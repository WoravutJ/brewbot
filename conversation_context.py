"""รวมเงื่อนไขสนทนาโดยเก็บ state แยกสำหรับผู้ใช้แต่ละคน."""

import json
import re
from dataclasses import dataclass, replace
from typing import Sequence

from guardrails import inspect_input
from query_parser import QueryPlan, normalize_query, parse_query


@dataclass(frozen=True)
class ConversationState:
    # Caller เก็บ state ใน session; ไม่เก็บในตัวแปรกลางหรือ cache โมเดล
    menu_plan: QueryPlan | None = None
    topic_query: str = ""
    menu_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedTurn:
    plan: QueryPlan | None
    state: ConversationState
    used_context: bool
    blocked_reply: str | None = None


FIELD_QUESTIONS = {
    "temperature": "ต้องการเครื่องดื่มแบบร้อนหรือแบบเย็นคะ?",
    "caffeine_status": "ต้องการแบบมีคาเฟอีนหรือไม่มีคาเฟอีนคะ?",
    "milk_requirement": "ต้องการใส่นมหรือเลี่ยงนมคะ?",
}
HEALTH_QUESTION = "อยากเน้นหวานน้อย เลี่ยงคาเฟอีน หรือคุณสมบัติอื่นคะ?"


def _unique(values) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _clarification(current: QueryPlan, previous: QueryPlan | None, merged: QueryPlan) -> str | None:
    # ไม่ลืมคำถามที่ยังไม่ได้คำตอบ เช่น ถามคาเฟอีน แต่ลูกค้าตอบเพียงอุณหภูมิ
    questions = []
    for field, question in FIELD_QUESTIONS.items():
        if question in (current.clarification_question or ""):
            questions.append(question)
        elif previous and question in (previous.clarification_question or ""):
            if getattr(current, field) is None:
                questions.append(question)
    if "เมนูสุขภาพ" in merged.preferences:
        has_detail = (
            merged.caffeine_status is not None or merged.milk_requirement is not None
            or any("หวาน" in preference for preference in merged.preferences)
        )
        if not has_detail:
            questions.append(HEALTH_QUESTION)
    return " ".join(_unique(questions)) or None


def resolve_query(
    question: str, state: ConversationState | None = None, *, menu_names: Sequence[str] = (),
) -> ResolvedTurn:
    """ข้อความล่าสุดแทนค่าที่ระบุใหม่ และรักษาเงื่อนไขอื่นของหัวข้อเมนูเดิม."""
    state = state if state is not None else ConversationState()
    check = inspect_input(question)
    if not check.allow:
        return ResolvedTurn(None, state, False, check.reply)
    text = normalize_query(question)
    # เริ่มหัวข้อใหม่โดยไม่ติดข้อจำกัดจากบทสนทนาเก่า
    reset = re.search(r"^(?:เริ่มใหม่|เริ่มเรื่องใหม่|เปลี่ยนเรื่อง|\bstart over\b|\bnew topic\b)", text)
    if reset:
        state = ConversationState()
        remaining = text[reset.end():].lstrip(" ,:;.!ๆ")
        if not remaining:
            return ResolvedTurn(None, state, False, "เริ่มใหม่แล้วค่ะ ต้องการสอบถามเมนูหรือข้อมูลร้านเรื่องใดคะ?")
        text = remaining

    current = replace(parse_query(text), original_query=question)
    names = _unique(name for name in menu_names if normalize_query(name) in text)
    if (names or current.needs_clarification) and current.intent == "unknown":
        current = replace(current, intent="menu_question", scope="menu")
    # คำถามร้านไม่รับเงื่อนไขเครื่องดื่ม แต่ยังเก็บไว้สำหรับการกลับมาถามเมนู
    if current.intent in {"store_information", "greeting"}:
        return ResolvedTurn(current, state, False)
    if re.search(r"อัน(?:แรก|นี้|นั้น)|ตัว(?:แรก|นี้|นั้น)|เมนู(?:แรก|นี้|นั้น)|\b(?:that one|first one)\b", text) and not names:
        clarified = replace(current, clarification_question="หมายถึงเมนูชื่ออะไรคะ? กรุณาระบุชื่อเมนูค่ะ")
        return ResolvedTurn(clarified, state, False)
    if current.intent == "unknown":
        # ไม่ผูกข้อความที่ไม่เข้าใจเข้ากับเมนูเดิมโดยอัตโนมัติ
        return ResolvedTurn(current, state, False)

    previous = state.menu_plan
    if previous is None:
        return ResolvedTurn(current, ConversationState(current, text, names), False)

    preferences = list(previous.preferences)
    if any("หวาน" in preference for preference in current.preferences):
        preferences = [preference for preference in preferences if "หวาน" not in preference]
    if "ถามถึงทั้งแบบร้อนและแบบเย็น" in current.preferences:
        temperature = None
    else:
        temperature = current.temperature if current.temperature is not None else previous.temperature
    if current.temperature is not None:
        preferences = [p for p in preferences if p != "ถามถึงทั้งแบบร้อนและแบบเย็น"]
    exclusions = _unique(previous.excluded_categories + current.excluded_categories)
    if current.category is not None:
        # คำขอหมวดใหม่แทนการยกเว้นหมวดเดียวกันในอดีต
        exclusions = tuple(category for category in exclusions if category != current.category)
    category = current.category if current.category is not None else previous.category
    if category in current.excluded_categories:
        category = None
    merged = replace(
        current, temperature=temperature,
        caffeine_status=current.caffeine_status if current.caffeine_status is not None else previous.caffeine_status,
        milk_requirement=current.milk_requirement if current.milk_requirement is not None else previous.milk_requirement,
        category=category, excluded_categories=exclusions,
        preferences=_unique(tuple(preferences) + current.preferences),
    )
    clarification = _clarification(current, previous, merged)
    # หมวดหรือชื่อเมนูใหม่เริ่มหัวข้อค้นหาใหม่ แต่คงข้อจำกัดอื่นของลูกค้า
    changed_value = any(
        getattr(current, field) is not None and getattr(previous, field) is not None
        and getattr(current, field) != getattr(previous, field)
        for field in ("temperature", "caffeine_status", "milk_requirement")
    )
    changed_sweetness = (
        any("หวาน" in p for p in current.preferences)
        and any("หวาน" in p for p in previous.preferences)
    )
    resolved_pending = previous.needs_clarification and clarification is None
    new_topic = bool(names) or current.category is not None or changed_value or changed_sweetness or resolved_pending
    topic = text if new_topic else state.topic_query
    focus_names = names if names or current.category is not None else state.menu_names
    parts = [text]
    if topic and topic != text:
        parts.append(f"หัวข้อเมนูก่อนหน้า: {topic}")
    for field, label in (
        ("temperature", "อุณหภูมิ"), ("caffeine_status", "คาเฟอีน"),
        ("category", "หมวด"), ("milk_requirement", "นม"),
    ):
        if getattr(merged, field) is not None:
            parts.append(f"{label}: {getattr(merged, field)}")
    if exclusions:
        parts.append("ไม่เอาหมวด: " + ", ".join(exclusions))
    if merged.preferences:
        parts.append("ความต้องการ: " + ", ".join(merged.preferences))
    # ชื่อเมนูเดิมที่ยืนยันได้ ใช้ช่วยค้นเมื่อถามต่อว่า 'ขอร้อน'
    if focus_names:
        parts.append("เมนูที่กำลังพูดถึง: " + ", ".join(focus_names))
    merged = replace(merged, normalized_query="\n".join(parts), clarification_question=clarification)
    return ResolvedTurn(merged, ConversationState(merged, topic, focus_names), True)


if __name__ == "__main__":
    # ทดสอบ state อย่างเดียว ไม่โหลด embedding และไม่เรียก Claude
    examples = (
        ("ไม่กินคาเฟอีน", "ขอร้อน ๆ", "เปลี่ยนเป็นเย็น", "ร้านเปิดกี่โมง", "ขอร้อน ๆ"),
        ("วันนี้อยากกินเมนูสุขภาพ", "หวานน้อย"),
        ("ขอร้อน มีคาเฟอีนและไม่มีคาเฟอีน", "ร้อน", "ไม่มีคาเฟอีน"),
    )
    for number, questions in enumerate(examples, start=1):
        print(f"\nบทสนทนาที่ {number}")
        state = ConversationState()
        for question in questions:
            turn = resolve_query(question, state)
            state = turn.state
            print(json.dumps({
                "question": question, "used_context": turn.used_context,
                "temperature": turn.plan.temperature if turn.plan else None,
                "caffeine_status": turn.plan.caffeine_status if turn.plan else None,
                "preferences": turn.plan.preferences if turn.plan else (),
                "clarification_question": turn.plan.clarification_question if turn.plan else turn.blocked_reply,
            }, ensure_ascii=False, indent=2))
    print("\nยังไม่ได้เชื่อม chat_service.py หรือหน้าเว็บ และยังไม่เรียก API")
