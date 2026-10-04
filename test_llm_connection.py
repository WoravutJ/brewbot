"""ทดสอบ Claude หนึ่งครั้งด้วยหลักฐานจาก Excel และจับเวลาจริง."""

from time import perf_counter

from knowledge_loader import KnowledgeDataError, load_menu
from llm_client import LLMClient, LLMClientError, check_configuration
from retriever import SearchResult, menu_text


def main() -> int:
    # ตรวจการตั้งค่าก่อน ไม่แสดง API key
    readiness = check_configuration()
    missing = [name for name, ready in readiness.items() if not ready]
    if missing:
        print("ยังไม่พร้อม: " + ", ".join(missing))
        print("ยังไม่เรียก API")
        return 1

    total_start = perf_counter()
    try:
        data_start = perf_counter()
        menu = load_menu()
        item = next((m for m in menu if m["name"] == "ลาเต้"), menu[0])
        question = f"{item['name']}ราคาเท่าไหร่"
        # เลือกหลักฐานตรงจากไฟล์ เพื่อทดสอบ API แยกจากการค้นหา
        evidence = [SearchResult(
            id=item["id"], kind="menu", source=item["source"],
            text=menu_text(item), score=0.0, metadata=dict(item),
        )]
        data_seconds = perf_counter() - data_start
        print(f"คำถาม: {question}")
        print(f"ราคาในไฟล์: {item['price']:g} บาท")
        print("กำลังเรียก Claude หนึ่งครั้ง…", flush=True)

        llm_start = perf_counter()
        reply = LLMClient().generate(question, evidence)
        llm_seconds = perf_counter() - llm_start
        total_seconds = perf_counter() - total_start

        print(f"\nคำตอบ: {reply.text}")
        print("รหัสหลักฐาน: " + ", ".join(reply.source_ids))
        print(f"เวลาอ่านและเตรียมข้อมูล: {data_seconds:.3f} วินาที")
        # รวมการเตรียม client, API และตรวจรูปแบบคำตอบ
        print(f"เวลา LLM client: {llm_seconds:.3f} วินาที")
        print(f"เวลารวมฝั่งโปรแกรม: {total_seconds:.3f} วินาที")
        print("เชื่อมต่อและตรวจรูปแบบสำเร็จ โปรดเทียบคำตอบกับราคาในไฟล์")
        print("ขั้นนี้ยังไม่ได้ทดสอบ embedding หรือ chatbot ทั้งระบบ")
        return 0
    except (KnowledgeDataError, LLMClientError) as exc:
        print(f"ทดสอบไม่สำเร็จ: {exc}")
        print(f"เวลาจนพบข้อผิดพลาด: {perf_counter() - total_start:.3f} วินาที")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
