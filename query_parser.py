"""แยกความต้องการจากคำถามด้วยกฎเบื้องต้น โดยไม่เรียก API."""

import json
import re
import unicodedata
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class QueryPlan:
    original_query: str
    normalized_query: str
    intent: str
    scope: str
    temperature: str | None
    caffeine_status: str | None
    category: str | None
    milk_requirement: str | None
    excluded_categories: tuple[str, ...]
    preferences: tuple[str, ...]
    clarification_question: str | None

    @property
    def needs_clarification(self) -> bool:
        return self.clarification_question is not None

    def to_dict(self) -> dict:
        return {**asdict(self), "needs_clarification": self.needs_clarification}


def normalize_query(text: str) -> str:
    # จัดรูปข้อความและแก้เฉพาะคำพิมพ์ผิดที่รู้จัก ไม่เดาคำทั่วไป
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text.replace("ห่ายง่ว", "หายง่วง")


def _matches(text: str, pattern: str) -> bool:
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _mentions(text: str, negative: str, positive: str) -> tuple[bool, bool]:
    # ตัดวลีปฏิเสธก่อน เพื่อไม่ให้ 'ไม่มีคาเฟอีน' ถูกอ่านเป็น 'มีคาเฟอีน'
    negatives = list(re.finditer(negative, text, flags=re.IGNORECASE))
    remaining = re.sub(negative, " ", text, flags=re.IGNORECASE)
    return bool(negatives), _matches(remaining, positive)


def parse_query(query: str) -> QueryPlan:
    """ตีความคำถามหนึ่งข้อความ; ยังไม่ตีความประวัติสนทนาหรือรับคำสั่งซื้อ."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("คำถามต้องเป็นข้อความที่ไม่ว่าง")
    text = normalize_query(query)
    temperature = caffeine = category = milk = None
    exclusions, preferences, questions = [], [], []

    no_caffeine, yes_caffeine = _mentions(
        text,
        r"(?:ไม่(?:กิน|เอา|ดื่ม|ต้องการ|รับ)|ไม่มี|งด|เลี่ยง|หลีกเลี่ยง)\s*คาเฟอีน"
        r"|\b(?:no|without)\s+caffeine\b|\bcaffeine[ -]?free\b|\bdecaf(?:feinated)?\b",
        r"มี\s*คาเฟอีน|(?:อยากได้|ต้องการ|ขอ)\s*คาเฟอีน"
        r"|\b(?:with\s+caffeine|caffeinated)\b",
    )
    # ถามคุณสมบัติของเมนู ไม่ได้แปลว่าขอเฉพาะเมนูที่มีคุณสมบัตินั้น
    if _matches(text, r"มี\s*คาเฟอีน\s*(?:ไหม|หรือเปล่า|มั้ย)|\b(?:does|is)\b.*\bcaffeinated\b"):
        yes_caffeine = False
    if no_caffeine and yes_caffeine:
        questions.append("ต้องการแบบมีคาเฟอีนหรือไม่มีคาเฟอีนคะ?")
    elif no_caffeine:
        caffeine = "ไม่มี"
    elif yes_caffeine:
        caffeine = "มี"

    no_milk, yes_milk = _mentions(
        text,
        r"(?:ไม่(?:กิน|เอา|ดื่ม|ต้องการ|ใส่)|ไม่มี|งด|แพ้)\s*นม"
        r"|\b(?:no|without)\s+milk\b|\b(?:dairy[ -]?free|milk[ -]?free|lactose[ -]?free)\b",
        r"ใส่นม|มีนม|นมเยอะ|\b(?:with\s+milk|milky|creamy)\b",
    )
    if _matches(text, r"(?:มี|ใส่)นม\s*(?:ไหม|หรือเปล่า|มั้ย)"):
        yes_milk = False
    # เก็บความต้องการเท่านั้น ยังไม่ใช้คำอธิบายเมนูรับรองส่วนผสมหรือสารก่อภูมิแพ้
    if no_milk and yes_milk:
        questions.append("ต้องการใส่นมหรือเลี่ยงนมคะ?")
    elif no_milk:
        milk = "เลี่ยงนมหรือส่วนประกอบที่ระบุ"
    elif yes_milk:
        milk = "มีนม"

    # อากาศร้อนไม่ได้แปลว่าลูกค้าสั่งเครื่องดื่มร้อน
    temperature_text = re.sub(r"อากาศ\s*ร้อน|วันนี้\s*ร้อน|\bhot\s+weather\b", " ", text)
    negative_temp = r"ไม่(?:เอา|กิน|ดื่ม|ต้องการ)\s*(?:แบบ)?\s*(?:ร้อน|เย็น)|\bnot\s+(?:hot|iced|cold)\b"
    if _matches(temperature_text, negative_temp):
        questions.append("ต้องการเครื่องดื่มแบบร้อนหรือแบบเย็นคะ?")
        temperature_text = re.sub(negative_temp, " ", temperature_text)
    hot = _matches(temperature_text, r"ร้อน|\bhot\b")
    cold = _matches(temperature_text, r"เย็น|\b(?:iced|cold)\b")
    if hot != cold and not _matches(text, negative_temp):
        temperature = "ร้อน" if hot else "เย็น"
    elif hot and cold:
        preferences.append("ถามถึงทั้งแบบร้อนและแบบเย็น")

    # ไม่เอากาแฟไม่ได้แปลว่าไม่มีคาเฟอีน เพราะลูกค้าอาจดื่มชาได้
    no_coffee = r"ไม่(?:เอา|กิน|ดื่ม|ต้องการ)\s*กาแฟ|\b(?:no|without)\s+coffee\b"
    category_text = text
    if _matches(text, no_coffee):
        exclusions.append("กาแฟ")
        category_text = re.sub(no_coffee, " ", category_text)
    category_patterns = {
        "กาแฟ": r"กาแฟ|ลาเต้|เอสเปรสโซ่|อเมริกาโน่|คาปูชิโน่|มอคค่า|\b(?:coffee|latte|espresso|americano|cappuccino|mocha)\b",
        "ชา": r"ชาไทย|ชาเขียว|มัทฉะ|(?:อยากได้|ขอ|ดื่ม)\s*ชา|\b(?:tea|matcha)\b",
        "เบเกอรี่": r"เบเกอรี่|ขนม|เค้ก|คุกกี้|ครัวซองต์|แซนวิช|\b(?:bakery|cake|cookie|croissant|sandwich)\b",
    }
    categories = [name for name, pattern in category_patterns.items() if _matches(category_text, pattern)]
    if len(categories) == 1 and categories[0] not in exclusions:
        category = categories[0]

    if _matches(text, r"หวานน้อย|ลดหวาน|\bless\s+sweet\b|\blow\s+sugar\b"):
        preferences.append("หวานน้อย")
    if _matches(text, r"ไม่หวาน|\bunsweetened\b|\bno\s+added\s+sugar\b"):
        preferences.append("ไม่หวานหรือไม่เติมน้ำตาลตามคำถาม")
    if _matches(text, r"สุขภาพ|\bhealthy\b"):
        preferences.append("เมนูสุขภาพ")
        if caffeine is None and milk is None and not any("หวาน" in p for p in preferences):
            questions.append("อยากเน้นหวานน้อย เลี่ยงคาเฟอีน หรือคุณสมบัติอื่นคะ?")
    if _matches(text, r"หายง่วง|ง่วง|ตื่นตัว|\bsleepy\b|\bstay\s+awake\b"):
        preferences.append("อยากรู้สึกตื่นตัว")  # ไม่บังคับคาเฟอีนหรือรับรองผล

    store_question = _matches(text, r"ร้านเปิด|ร้านปิด|เปิดกี่|ปิดกี่|ที่ตั้ง|ที่จอด|จอดรถ|ไวไฟ|wi[ -]?fi|board\s*game|โต๊ะ|จอง|คืนเงิน|ชำระ|\b(?:opening|hours|parking|reservation|refund|payment|location)\b|what\s+time.*\b(?:open|close)\b")
    # 'จองโต๊ะพรุ่งนี้เย็น' ใช้เย็นเป็นเวลา ไม่ใช่อุณหภูมิเครื่องดื่ม
    explicit_menu = _matches(text, r"เมนู|เครื่องดื่ม|กาแฟ|ชาไทย|ชาเย็น|ชาเขียว|ลาเต้|ขนม|\b(?:menu|drink|coffee|tea|latte)\b")
    if store_question and not explicit_menu and category is None and caffeine is None and milk is None and not preferences:
        temperature = None
    menu_question = category is not None or caffeine is not None or milk is not None or temperature is not None or bool(preferences) or bool(exclusions) or _matches(text, r"เมนู|เครื่องดื่ม|ราคา|แนะนำ|อยากกิน|\b(?:menu|drink|recommend|price)\b")
    if store_question and menu_question:
        intent, scope = "mixed_question", "all"
    elif store_question:
        intent, scope = "store_information", "documents"
    elif menu_question:
        intent, scope = "menu_question", "menu"
    elif _matches(text, r"^(?:สวัสดี(?:ครับ|ค่ะ|คะ)?|hello|hi)[! .]*$"):
        intent, scope = "greeting", "all"
    else:
        intent, scope = "unknown", "all"

    return QueryPlan(query, text, intent, scope, temperature, caffeine, category, milk,
                     tuple(exclusions), tuple(preferences), " ".join(questions) or None)


if __name__ == "__main__":
    # ทดสอบการแยกคำถามอย่างเดียว ไม่โหลด embedding และไม่เรียก Claude
    examples = [
        "ไม่กินคาเฟอีน ขอร้อน ๆ",
        "วันนี้อยากกินเมนูสุขภาพ",
        "อยากหายง่วง แต่ไม่กินคาเฟอีน",
        "อยากได้กาแฟรสนุ่มใส่นม",
        "ไม่เอากาแฟ ขอชาเย็น",
        "ร้านเปิดกี่โมง",
    ]
    for question in examples:
        print(json.dumps(parse_query(question).to_dict(), ensure_ascii=False, indent=2))
