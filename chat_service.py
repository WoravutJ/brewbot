"""เชื่อม RAG กับ Claude พร้อมบริบท สถานะ และเวลาของแต่ละข้อความ."""

import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Iterator

from config import MAX_INPUT_CHARS, TOP_K
from conversation_context import ConversationState, resolve_query
from guardrails import inspect_input, inspect_output, inspect_recommendations
from llm_client import LLMClient, LLMClientError
from query_parser import QueryPlan, normalize_query
from retriever import Retriever, SearchFilters, SearchResult


@dataclass(frozen=True)
class ChatResult:
    text: str
    reason: str
    recommended_menu_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    timings: dict[str, float]
    total_seconds: float
    context_state: ConversationState = field(default_factory=ConversationState)
    used_context: bool = False
    resolved_plan: QueryPlan | None = None


@dataclass(frozen=True)
class ChatEvent:
    # UI แสดง status ระหว่างรอ และแสดง result เมื่อเสร็จ
    status: str
    result: ChatResult | None = None


@contextmanager
def _measure(timings: dict, name: str):
    started = perf_counter()
    try:
        yield
    finally:
        timings[name] = perf_counter() - started


class ChatService:
    def __init__(self, *, retriever=None, llm=None, min_score: float | None = None):
        # โหลดโมเดลครั้งเดียวเมื่อเปิดระบบ ไม่โหลดใหม่ทุกคำถาม
        started = perf_counter()
        self.retriever = retriever if retriever is not None else Retriever()
        self.llm = llm if llm is not None else LLMClient()
        self.min_score = min_score  # ตั้งเกณฑ์ได้หลังประเมินชุดทดสอบจริง
        self.startup_seconds = perf_counter() - started

    def _retrieve(self, plan: QueryPlan) -> list[SearchResult]:
        filters = SearchFilters(plan.temperature, plan.caffeine_status, plan.category)
        menu_results, document_results = [], []
        if plan.scope in {"menu", "all"}:
            # ค้นผู้สมัครทั้งหมด เพื่อไม่ให้ผลที่ถูกตัดเรื่องนมบังเมนูอันดับถัดไป
            count = max(1, len(self.retriever.records))
            candidates = self.retriever.search(
                plan.normalized_query, scope="menu", filters=filters,
                top_k=count, min_score=self.min_score,
            )
            menu_results = [
                item for item in candidates
                if inspect_recommendations([item.metadata], plan).allow
            ]

            # ถ้าถามชื่อเมนูชัดเจน ต้องไม่ใช้เมนูอื่นตอบแทน
            named_ids = {
                record["id"] for record in self.retriever.records
                if record["kind"] == "menu"
                and normalize_query(record["metadata"]["name"]) in plan.normalized_query
            }
            if named_ids:
                menu_results = [item for item in menu_results if item.id in named_ids]
            menu_results = menu_results[:TOP_K]

        if plan.scope in {"documents", "all"}:
            # คำถามผสมยังค้นเอกสารร้านได้ แม้มีเงื่อนไขเมนู
            document_results = self.retriever.search(
                plan.normalized_query, scope="documents", top_k=TOP_K,
                min_score=self.min_score,
            )
        return menu_results + document_results

    def stream(
        self, question: str, *, state: ConversationState | None = None,
    ) -> Iterator[ChatEvent]:
        """รับและคืน state; ไม่เก็บบริบทผู้ใช้ไว้ใน service ที่ใช้ร่วมกัน."""
        started = perf_counter()
        next_state = state if state is not None else ConversationState()
        used_context, plan = False, None
        timings = {name: 0.0 for name in (
            "input_guard", "query_parser", "retrieval", "llm_client", "output_guard",
        )}

        def finish(text, reason, menu_ids=(), source_ids=()):
            return ChatEvent("เสร็จแล้ว", ChatResult(
                text, reason, tuple(menu_ids), tuple(source_ids),
                dict(timings), perf_counter() - started,
                next_state, used_context, plan,
            ))

        yield ChatEvent("กำลังตรวจสอบคำถาม…")
        with _measure(timings, "input_guard"):
            check = inspect_input(question)
        if not check.allow:
            yield finish(check.reply, check.reason)
            return

        yield ChatEvent("กำลังวิเคราะห์ความต้องการ…")
        with _measure(timings, "query_parser"):
            menu_names = tuple(
                record["metadata"]["name"] for record in self.retriever.records
                if record["kind"] == "menu"
            )
            turn = resolve_query(question, next_state, menu_names=menu_names)
            plan, next_state, used_context = turn.plan, turn.state, turn.used_context
        if plan is None:
            yield finish(turn.blocked_reply or "กรุณาระบุคำถามค่ะ", "context_reset")
            return
        if plan.needs_clarification:
            yield finish(plan.clarification_question, "clarification_required")
            return
        if plan.intent == "greeting":
            yield finish("สวัสดีค่ะ ต้องการสอบถามเมนูหรือข้อมูลร้านเรื่องใดคะ?", "greeting")
            return
        if plan.milk_requirement not in {None, "มีนม"}:
            # ยังไม่มีสูตรและข้อมูลสารก่อภูมิแพ้ครบ จึงส่งให้พนักงานตรวจ
            yield finish(
                "กรุณาตรวจสอบส่วนผสมกับพนักงานก่อนเลือกเมนูที่ต้องเลี่ยงนมหรือส่วนประกอบค่ะ",
                "milk_avoidance_unverified",
            )
            return

        yield ChatEvent("กำลังค้นหาเมนูและข้อมูลร้าน…")
        try:
            with _measure(timings, "retrieval"):
                evidence = self._retrieve(plan)
        except Exception:
            yield finish("ค้นหาข้อมูลไม่สำเร็จ กรุณาลองใหม่ค่ะ", "retrieval_error")
            return
        if not evidence:
            yield finish("ไม่พบข้อมูลที่ตรงคำถามหรือเงื่อนไขในข้อมูลร้านค่ะ", "no_evidence")
            return

        llm_question = question
        if used_context:
            # ส่งเฉพาะหัวข้อและเงื่อนไขลูกค้า ไม่ใช้คำตอบเก่าเป็นข้อเท็จจริงร้าน
            context = {
                "topic": next_state.topic_query,
                "menu_names": next_state.menu_names,
                "temperature": plan.temperature,
                "caffeine_status": plan.caffeine_status,
                "category": plan.category,
                "milk_requirement": plan.milk_requirement,
                "excluded_categories": plan.excluded_categories,
                "preferences": plan.preferences,
            }
            llm_question = (
                f"คำถามล่าสุด: {question}\n"
                "บริบทและเงื่อนไขปัจจุบันที่ลูกค้าระบุ: "
                + json.dumps(context, ensure_ascii=False)
            )
        if len(llm_question) > MAX_INPUT_CHARS:
            yield finish("คำถามรวมบริบทยาวเกินขนาดที่รองรับ กรุณาถามให้สั้นลงค่ะ", "context_too_long")
            return

        yield ChatEvent("กำลังเตรียมคำตอบ…")
        try:
            with _measure(timings, "llm_client"):
                reply = self.llm.generate(llm_question, evidence)
        except LLMClientError as exc:
            yield finish(str(exc), "llm_error")
            return
        except Exception:
            yield finish("เตรียมคำตอบไม่สำเร็จ กรุณาลองใหม่ค่ะ", "llm_error")
            return

        yield ChatEvent("กำลังตรวจสอบคำตอบ…")
        with _measure(timings, "output_guard"):
            by_id = {item.id: item for item in evidence}
            allowed_menu = {item.id for item in evidence if item.kind == "menu"}
            valid_ids = (
                bool(reply.source_ids)
                and set(reply.source_ids) <= set(by_id)
                and set(reply.recommended_menu_ids) <= allowed_menu
                and set(reply.recommended_menu_ids) <= set(reply.source_ids)
            )
            check = inspect_output(reply.text)
            if valid_ids and check.allow and reply.recommended_menu_ids:
                check = inspect_recommendations(
                    [by_id[item_id].metadata for item_id in reply.recommended_menu_ids], plan,
                )
        if not valid_ids:
            yield finish("ยังยืนยันแหล่งข้อมูลของคำตอบไม่ได้ค่ะ", "invalid_source_ids")
        elif not check.allow:
            yield finish(check.reply, check.reason)
        else:
            # ตรวจรหัสและเงื่อนไขได้ แต่ยังต้องประเมินความถูกต้องของข้อความจริง
            yield finish(reply.text, "answered", reply.recommended_menu_ids, reply.source_ids)

    def answer(self, question: str, *, state: ConversationState | None = None) -> ChatResult:
        """ใช้กับชุดทดสอบที่ไม่ต้องแสดงสถานะระหว่างทาง."""
        for event in self.stream(question, state=state):
            if event.result is not None:
                return event.result
        raise RuntimeError("Chat service ไม่ได้ส่งผลลัพธ์สุดท้าย")


def main() -> int:
    # --conversation รับแต่ละข้อความแยกกัน และใช้โมเดลที่โหลดไว้ซ้ำ
    args = sys.argv[1:]
    if args and args[0] == "--conversation":
        questions = args[1:]
        if not questions:
            print('ใช้: python3 chat_service.py --conversation "ไม่กินคาเฟอีน" "ขอร้อน ๆ"')
            return 1
    else:
        questions = [" ".join(args) or "ไม่กินคาเฟอีน ขอร้อน ๆ"]
    print("กำลังโหลดข้อมูลและ embedding model…", flush=True)
    try:
        service = ChatService()
    except Exception:
        print("เปิดระบบไม่สำเร็จ กรุณารัน knowledge_loader.py และ retriever.py เพื่อตรวจข้อมูล")
        return 1
    print(f"เวลาเปิดระบบ: {service.startup_seconds:.3f} วินาที (แยกจากเวลาตอบ)")
    state, exit_code = ConversationState(), 0
    for question in questions:
        print(f"\nคำถาม: {question}")
        result = None
        for event in service.stream(question, state=state):
            if event.result is None:
                print(event.status, flush=True)
            else:
                result = event.result
        state = result.context_state
        print(f"\nคำตอบ: {result.text}")
        print(f"ผล: {result.reason} / ใช้บริบท: {result.used_context}")
        if result.resolved_plan:
            print(f"เงื่อนไข: คาเฟอีน={result.resolved_plan.caffeine_status} / อุณหภูมิ={result.resolved_plan.temperature}")
        print("เมนูที่แนะนำ: " + (", ".join(result.recommended_menu_ids) or "—"))
        print("รหัสหลักฐาน: " + (", ".join(result.source_ids) or "—"))
        for name, seconds in result.timings.items():
            print(f"  {name}: {seconds:.3f} วินาที")
        print(f"เวลารวมฝั่งโปรแกรม: {result.total_seconds:.3f} วินาที")
        if result.reason in {"retrieval_error", "llm_error", "invalid_source_ids"}:
            exit_code = 1
    print("เวลา llm_client รวม API และการตรวจรูปแบบ; ยังไม่รวมเวลาแสดงผลในเว็บ")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
