"""หน้าแชต Streamlit ของ Brew & Book พร้อมสถานะและเวลาตอบ."""

import os
from dataclasses import asdict

import streamlit as st


STYLE = """
<style>
[data-testid="stAppViewContainer"] {background: #faf7f2; color: #29352d;
    --text-color: #29352d; --background-color: #faf7f2;
    --secondary-background-color: #eee6da; color-scheme: light;}
[data-testid="stSidebar"] {background: #eee6da; color: #29352d;}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stCaptionContainer"] {color: #29352d;}
[data-testid="stChatInput"], [data-testid="stBottom"] {background: #faf7f2;}
[data-testid="stChatInput"] textarea {background: #fffdf9; color: #29352d;
    -webkit-text-fill-color: #29352d; caret-color: #29352d;}
[data-testid="stChatInput"] textarea::placeholder {color: #667267;
    -webkit-text-fill-color: #667267;}
[data-testid="stExpander"], [data-testid="stTable"] {background: #fffdf9; color: #29352d;}
[data-testid="stExpander"] summary,
[data-testid="stTable"] th, [data-testid="stTable"] td {color: #29352d;}
.block-container {max-width: 880px; padding-top: 2.2rem;}
.brew-hero {padding: 26px 30px; border-radius: 24px; margin-bottom: 22px;
    background: linear-gradient(120deg, #354c3e, #526e55); color: #fffaf1;}
.brew-hero h1 {color: #fffaf1 !important; margin: 0; font-size: 2.3rem;}
.brew-hero p {margin: 8px 0 0; color: #eee7d9 !important;}
.brew-label {letter-spacing: .15em; font-size: .75rem; opacity: .85;}
[data-testid="stChatMessage"] {border-radius: 18px; border: 1px solid #e7dfd3;
    background: #fffdf9; margin-bottom: 12px;}
.stButton > button {border-radius: 14px; border-color: #d9cdbb;
    background: #fffdf9; color: #29352d;}
</style>
"""


def _load_secrets() -> None:
    # ใช้ environment บน Mac และรองรับ Secrets เมื่อ deploy
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "BREWBOT_DATA_DIR"):
        if os.getenv(name, "").strip():
            continue
        try:
            value = st.secrets.get(name)
        except FileNotFoundError:
            continue  # เครื่องที่ใช้ environment ไม่จำเป็นต้องมี secrets.toml
        except Exception:
            st.error("อ่านการตั้งค่า Secrets ไม่สำเร็จ กรุณาตรวจการตั้งค่าระบบ")
            st.stop()
        if isinstance(value, str) and value.strip():
            os.environ[name] = value.strip()


@st.cache_resource(show_spinner=False, max_entries=2)
def _get_service(model: str, data_stamp: tuple):
    # cache เฉพาะระบบและโมเดล ไม่เก็บข้อความลูกค้าใน cache กลาง
    from chat_service import ChatService
    from llm_client import LLMClient
    return ChatService(llm=LLMClient(model=model))


def _render_message(message: dict, show_details: bool) -> None:
    role = message["role"]
    with st.chat_message(role, avatar="☕" if role == "assistant" else "👤"):
        # แสดงเป็นข้อความ ไม่อนุญาต HTML จากลูกค้าหรือโมเดล
        st.write(message["content"])
        result = message.get("result")
        if result:
            st.caption(f"เวลาตอบฝั่งระบบ {result['total_seconds']:.3f} วินาที")
            if show_details:
                with st.expander("รายละเอียดการทดสอบ"):
                    labels = {
                        "input_guard": "ตรวจคำถาม",
                        "query_parser": "วิเคราะห์ความต้องการ",
                        "retrieval": "ค้นหา RAG",
                        "llm_client": "Claude และตรวจรูปแบบ",
                        "output_guard": "ตรวจคำตอบขั้นสุดท้าย",
                    }
                    st.table([
                        {"ขั้นตอน": labels.get(name, name), "วินาที": round(seconds, 4)}
                        for name, seconds in result["timings"].items()
                    ])
                    st.write("ผล:", result["reason"])
                    st.write("เมนูที่แนะนำ:", ", ".join(result["recommended_menu_ids"]) or "—")
                    st.write("รหัสหลักฐาน:", ", ".join(result["source_ids"]) or "—")
                    st.write("ใช้บริบทก่อนหน้า:", "ใช่" if result.get("used_context") else "ไม่")
                    plan = result.get("resolved_plan")
                    if plan:
                        st.write("เงื่อนไขคาเฟอีน:", plan.get("caffeine_status") or "ไม่ได้ระบุ")
                        st.write("อุณหภูมิ:", plan.get("temperature") or "ไม่ได้ระบุ")


def main() -> None:
    st.set_page_config(page_title="Brew & Book · BrewBot", page_icon="☕", layout="centered")
    st.markdown(STYLE, unsafe_allow_html=True)  # HTML คงที่ที่เขียนไว้ในแอปเท่านั้น
    _load_secrets()

    # import หลังตั้ง Secrets เพื่อให้ config อ่านค่าที่พร้อมแล้ว
    from config import MAX_INPUT_CHARS, MENU_FILE, STORE_INFO_FILE, STORE_POLICIES_FILE
    from llm_client import check_configuration
    from conversation_context import ConversationState

    st.markdown(
        '<div class="brew-hero"><div class="brew-label">YOUR CAFÉ COMPANION</div>'
        '<h1>☕ Brew &amp; Book</h1><p>คุยกับบรูว์ เพื่อค้นหาเมนูและข้อมูลร้าน</p></div>',
        unsafe_allow_html=True,
    )
    with st.sidebar:
        st.title("BrewBot")
        st.write("ผู้ช่วยเลือกเมนูและตอบคำถามเกี่ยวกับร้าน")
        show_details = st.toggle("แสดงรายละเอียดเวลาเพื่อทดสอบ", value=False)
        if st.button("เริ่มบทสนทนาใหม่", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_state = ConversationState()
            st.rerun()
        st.caption("เวลาตอบที่แสดงไม่รวมเครือข่ายระหว่างเว็บกับเครื่องผู้ใช้")

    readiness = check_configuration()
    if not all(readiness.values()):
        st.error("ระบบยังตั้งค่าไม่ครบ กรุณาตรวจด้วย llm_client.py ใน Terminal ที่ใช้เปิดเว็บ")
        st.stop()
    try:
        # เมื่อไฟล์ข้อมูลเปลี่ยน ให้สร้างดัชนีใหม่ในรอบที่เปิดหน้าเว็บถัดไป
        data_stamp = tuple(
            (str(path), path.stat().st_mtime_ns, path.stat().st_size)
            for path in (MENU_FILE, STORE_INFO_FILE, STORE_POLICIES_FILE)
        )
        with st.spinner("กำลังเตรียมข้อมูลและโมเดล ครั้งแรกอาจใช้เวลาสักครู่…"):
            service = _get_service(os.environ["ANTHROPIC_MODEL"].strip(), data_stamp)
    except Exception:
        st.error("เตรียมระบบไม่สำเร็จ กรุณาตรวจ knowledge_loader.py และ retriever.py ใน Terminal")
        st.stop()

    with st.sidebar:
        st.success("พร้อมให้บริการ")
        if show_details:
            st.caption(f"เวลาโหลดระบบครั้งแรก: {service.startup_seconds:.3f} วินาที")

    if "messages" not in st.session_state:
        st.session_state.messages = []  # ประวัติหน้าจอแยกตาม session
    if "conversation_state" not in st.session_state:
        st.session_state.conversation_state = ConversationState()
    st.caption("ถามต่อได้ เช่น ‘ไม่กินคาเฟอีน’ แล้ว ‘ขอร้อน ๆ’ บรูว์จะจำเงื่อนไขให้ค่ะ")
    st.caption("ต้องการล้างเงื่อนไขเดิม กด ‘เริ่มบทสนทนาใหม่’ หรือพิมพ์ ‘เริ่มใหม่’")

    selected = None
    if not st.session_state.messages:
        st.write("เริ่มจากคำถามที่คุณสนใจ")
        examples = (
            "ไม่กินคาเฟอีน ขอร้อน ๆ", "อยากได้กาแฟรสนุ่มใส่นม",
            "วันนี้อยากกินเมนูสุขภาพ", "ร้านเปิดกี่โมง",
        )
        columns = st.columns(2)
        for index, example in enumerate(examples):
            if columns[index % 2].button(example, key=f"example_{index}", use_container_width=True):
                selected = example

    for message in st.session_state.messages:
        _render_message(message, show_details)
    prompt = st.chat_input("สอบถามเมนูหรือข้อมูลร้าน…", max_chars=MAX_INPUT_CHARS)
    question = selected or prompt
    if not question:
        return

    user_message = {"role": "user", "content": question}
    st.session_state.messages.append(user_message)
    _render_message(user_message, show_details)
    result = None
    try:
        with st.status("กำลังตรวจสอบคำถาม…", expanded=False) as status:
            for event in service.stream(question, state=st.session_state.conversation_state):
                if event.result is None:
                    status.update(label=event.status, state="running")
                else:
                    result = event.result
            if result is None:
                raise RuntimeError("ไม่มีผลลัพธ์สุดท้าย")
            # เก็บบริบทเฉพาะ session นี้ ไม่เก็บใน service ที่ cache ใช้ร่วมกัน
            st.session_state.conversation_state = result.context_state
            failed = result.reason in {"llm_error", "retrieval_error", "invalid_source_ids"}
            status.update(
                label="พบปัญหาระหว่างประมวลผล" if failed else "เตรียมคำตอบเรียบร้อยแล้ว",
                state="error" if failed else "complete", expanded=False,
            )
        message = {"role": "assistant", "content": result.text, "result": asdict(result)}
    except Exception:
        # ไม่แสดง traceback หรือรายละเอียดคำขอในหน้าลูกค้า
        message = {"role": "assistant", "content": "ระบบมีปัญหาระหว่างตอบ กรุณาลองใหม่ค่ะ"}
    st.session_state.messages.append(message)
    st.session_state.messages = st.session_state.messages[-40:]
    _render_message(message, show_details)


if __name__ == "__main__":
    main()
