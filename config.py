"""ค่าตั้งค่ากลางของ Brew Bot ใช้ร่วมกันใน Colab และ HF Spaces."""

import os
from pathlib import Path


# หาโฟลเดอร์จากตำแหน่งไฟล์ ไม่ขึ้นกับโฟลเดอร์ที่สั่งรัน
PROJECT_DIR = Path(__file__).resolve().parent

# เปลี่ยนที่เก็บข้อมูลได้ผ่าน environment variable
_data_dir = os.getenv("BREWBOT_DATA_DIR", "").strip()
DATA_DIR = Path(_data_dir).expanduser() if _data_dir else PROJECT_DIR / "data"
if not DATA_DIR.is_absolute():
    DATA_DIR = PROJECT_DIR / DATA_DIR
DATA_DIR = DATA_DIR.resolve()

# ชื่อไฟล์ข้อมูลภายนอกและ sheet ที่ระบบจะอ่าน
MENU_FILE = DATA_DIR / "menu.xlsx"
MENU_SHEET = "Menu"
STORE_INFO_FILE = DATA_DIR / "store_info.md"
STORE_POLICIES_FILE = DATA_DIR / "store_policies.md"

# ใช้ embedding model เดียวกับ V4 เพื่อเริ่มเปรียบเทียบผล
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
TOP_K = 5

# กำหนดชื่อโมเดลเมื่อถึงขั้นเชื่อม Claude
LLM_MODEL = os.getenv("ANTHROPIC_MODEL", "").strip()
API_KEY_ENV = "ANTHROPIC_API_KEY"  # เก็บค่าจริงใน Secrets หรือ environment
MAX_RESPONSE_TOKENS = 700
API_TIMEOUT_SECONDS = 30.0

# จำกัดข้อความ ประวัติที่ส่งให้โมเดล และจำนวนรอบเรียก tool
MAX_INPUT_CHARS = 2000
MAX_OUTPUT_CHARS = 3000
MAX_HISTORY_MESSAGES = 12
MAX_TOOL_ROUNDS = 3


if __name__ == "__main__":
    # แสดงตำแหน่งข้อมูลเท่านั้น ยังไม่โหลดข้อมูลหรือเรียก API
    print("Brew Bot configuration ready")
    print(f"Data folder: {DATA_DIR}")
    for file_path in (MENU_FILE, STORE_INFO_FILE, STORE_POLICIES_FILE):
        status = "พบไฟล์" if file_path.is_file() else "ยังไม่พบไฟล์"
        print(f"  {file_path.name}: {status}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Claude model: {LLM_MODEL or 'ยังไม่กำหนด (ทำในขั้นเชื่อม Claude)'}")
