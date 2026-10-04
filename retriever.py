"""ค้นหาความหมายด้วย embedding และกรองคุณสมบัติจากข้อมูลร้าน."""

from dataclasses import dataclass
import math

import numpy as np

from config import EMBEDDING_MODEL, TOP_K
from knowledge_loader import KnowledgeBase, load_knowledge_base


@dataclass(frozen=True)
class SearchFilters:
    # None หมายถึงลูกค้าไม่ได้ระบุเงื่อนไขนั้น
    temperature: str | None = None
    caffeine_status: str | None = None
    category: str | None = None


@dataclass(frozen=True)
class SearchResult:
    id: str
    kind: str
    source: str
    text: str
    score: float
    metadata: dict


def menu_text(item: dict) -> str:
    """สร้างข้อความจากข้อมูลปัจจุบัน ไม่เก็บสำเนาราคาในโค้ด."""
    return (
        f"เมนู: {item['name']}\nหมวด: {item['category']}\n"
        f"ราคา: {item['price']:g} บาท\nอุณหภูมิ: {item['temperatures']}\n"
        f"คาเฟอีน: {item['caffeine_status']}\nความหวาน: {item['sweetness']}\n"
        f"ตัวเลือกความหวาน: {item['sweetness_options']}\n"
        f"คำอธิบาย: {item['description']}"
    )


def _unit_vectors(vectors: np.ndarray) -> np.ndarray:
    """ปรับความยาวเวกเตอร์โดยไม่ใช้ matmul และป้องกัน overflow."""
    if not np.isfinite(vectors).all():
        raise ValueError("Embedding มี NaN หรือ infinity")
    scale = np.max(np.abs(vectors), axis=1, keepdims=True)
    if np.any(scale == 0):
        raise ValueError("Embedding เป็นเวกเตอร์ศูนย์")
    scaled = vectors / scale
    lengths = np.sqrt(np.sum(scaled * scaled, axis=1, keepdims=True))
    return scaled / lengths


class Retriever:
    def __init__(self, knowledge: KnowledgeBase | None = None, *, encoder=None):
        # เปิดช่องให้ใช้ encoder จำลองในการทดสอบโดยไม่ดาวน์โหลดโมเดล
        if encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "ติดตั้งก่อน: python3 -m pip install sentence-transformers"
                ) from exc
            encoder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
        self.encoder = encoder
        self.reload(knowledge)

    def reload(self, knowledge: KnowledgeBase | None = None) -> None:
        """โหลดข้อมูลและทำ embedding ใหม่เมื่อข้อมูลร้านเปลี่ยน."""
        knowledge = knowledge if knowledge is not None else load_knowledge_base()
        records = []
        for item in knowledge.menu:
            records.append({
                "id": item["id"], "kind": "menu", "source": item["source"],
                "text": menu_text(item), "metadata": dict(item),
            })
        for document in knowledge.documents:
            records.append({
                "id": document.id, "kind": "document", "source": document.source,
                "text": document.text, "metadata": {"title": document.title},
            })
        if not records:
            raise ValueError("ไม่มีข้อมูลสำหรับสร้างดัชนีค้นหา")

        # E5 ใช้ passage: สำหรับเอกสาร และ query: สำหรับคำถาม
        vectors = np.asarray(self.encoder.encode(
            [f"passage: {r['text']}" for r in records],
            normalize_embeddings=True, show_progress_bar=False,
        ), dtype=np.float64)
        if vectors.ndim != 2 or vectors.shape[0] != len(records) or vectors.shape[1] == 0:
            raise ValueError("จำนวนหรือรูปแบบ embedding ไม่ตรงกับข้อมูล")
        normalized = _unit_vectors(vectors)
        # เปลี่ยนดัชนีหลังสร้างสำเร็จทั้งหมด ป้องกันข้อมูลกับเวกเตอร์ไม่ตรงกัน
        self.records, self.vectors = records, normalized

    def search(
        self, query: str, *, filters: SearchFilters | None = None,
        scope: str = "all", top_k: int = TOP_K, min_score: float | None = None,
    ) -> list[SearchResult]:
        """คืนหลักฐานที่ใกล้เคียง ไม่ใช่คำตอบสุดท้ายหรือค่าความมั่นใจ."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("คำค้นหาต้องเป็นข้อความที่ไม่ว่าง")
        if scope not in {"all", "menu", "documents"}:
            raise ValueError("scope ต้องเป็น all / menu / documents")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k ต้องเป็นจำนวนเต็มตั้งแต่ 1")
        if min_score is not None and (not math.isfinite(min_score) or not -1 <= min_score <= 1):
            raise ValueError("min_score ต้องอยู่ระหว่าง -1 และ 1")
        filters = filters or SearchFilters()
        if filters.temperature not in {None, "ร้อน", "เย็น"}:
            raise ValueError("เงื่อนไขอุณหภูมิต้องเป็น ร้อน / เย็น")
        if filters.caffeine_status not in {None, "มี", "ไม่มี"}:
            raise ValueError("เงื่อนไขคาเฟอีนต้องเป็น มี / ไม่มี")
        if filters.category is not None and not filters.category.strip():
            raise ValueError("เงื่อนไขหมวดเมนูต้องไม่ว่าง")
        has_filters = any(v is not None for v in (
            filters.temperature, filters.caffeine_status, filters.category,
        ))
        if has_filters and scope == "documents":
            raise ValueError("เงื่อนไขเมนูใช้กับ scope documents ไม่ได้")

        # กรองก่อนจัดอันดับ: ไม่ให้เอกสารหรือเมนูไม่ทราบคุณสมบัติผ่านข้อห้าม
        indices = []
        for i, record in enumerate(self.records):
            is_menu = record["kind"] == "menu"
            if (scope == "menu" or has_filters) and not is_menu:
                continue
            if scope == "documents" and is_menu:
                continue
            if is_menu:
                item = record["metadata"]
                if filters.temperature is not None:
                    options = item["temperature_options"]
                    if options is None or filters.temperature not in options:
                        continue
                if filters.caffeine_status is not None and item["caffeine_status"] != filters.caffeine_status:
                    continue
                if filters.category is not None and item["category"] != filters.category:
                    continue
            indices.append(i)
        if not indices:
            return []

        query_vectors = np.asarray(self.encoder.encode(
            [f"query: {query.strip()}"], normalize_embeddings=True,
            show_progress_bar=False,
        ), dtype=np.float64)
        if query_vectors.shape != (1, self.vectors.shape[1]):
            raise ValueError("Embedding ของคำถามมีมิติไม่ตรงกับดัชนี")
        vector = _unit_vectors(query_vectors)[0]
        # ใช้คูณรายตำแหน่งแล้วรวม หลีกเลี่ยง matmul warning บน Mac บางชุดติดตั้ง
        scores = np.sum(self.vectors[indices] * vector, axis=1, dtype=np.float64)
        if not np.isfinite(scores).all():
            raise ValueError("คะแนนค้นหามี NaN หรือ infinity")
        if np.any(np.abs(scores) > 1.0 + 1e-10):
            raise ValueError("คะแนน cosine similarity อยู่นอกช่วงที่ถูกต้อง")
        scores = np.clip(scores, -1.0, 1.0)  # แก้ความคลาดเคลื่อนทศนิยมเล็กน้อย
        ranked = np.argsort(-scores, kind="stable")
        results = []
        for position in ranked:
            score = float(scores[position])
            # ไม่มี fallback ที่นำผลต่ำกว่าเกณฑ์กลับมาอีก
            if min_score is not None and score < min_score:
                continue
            record = self.records[indices[position]]
            results.append(SearchResult(
                record["id"], record["kind"], record["source"], record["text"],
                score, dict(record["metadata"]),
            ))
            if len(results) == top_k:
                break
        return results


if __name__ == "__main__":
    print("กำลังโหลดข้อมูลและ embedding model; ครั้งแรกต้องใช้อินเทอร์เน็ต")
    retriever = Retriever()
    # ตอนนี้ส่งเงื่อนไขชัดเจนให้ทดสอบ ส่วนแยกภาษาลูกค้าจะทำในไฟล์ถัดไป
    examples = [
        ("อยากได้กาแฟรสนุ่มใส่นม", "menu", SearchFilters()),
        ("ไม่กินคาเฟอีน ขอร้อน ๆ", "menu", SearchFilters("ร้อน", "ไม่มี")),
        ("ร้านเปิดกี่โมง", "documents", SearchFilters()),
    ]
    for query, scope, filters in examples:
        print(f"\nคำถาม: {query}")
        for result in retriever.search(query, scope=scope, filters=filters, top_k=3):
            label = result.metadata.get("name") or result.metadata.get("title")
            print(f"  {result.id} {label} (similarity={result.score:.3f})")
    print("\nผลนี้เป็นรายการค้นหา ยังไม่ใช่คำตอบของ chatbot และยังไม่เรียก Claude")
