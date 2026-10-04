"""ตรวจรูปและเตรียม image block สำหรับ Claude โดยไม่เรียก API."""

from __future__ import annotations

import base64
import hashlib
import io
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, UnidentifiedImageError

# จำกัดต่อรูปเพื่อควบคุมหน่วยความจำและขนาดคำขอ
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_IMAGE_SIDE = 8000
MAX_IMAGE_PIXELS = 20_000_000
SUPPORTED_FORMATS = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}


class ImageValidationError(ValueError):
    """ข้อความที่แสดงให้ลูกค้าได้โดยไม่เปิดเผยรายละเอียดระบบ."""


@dataclass(frozen=True)
class PreparedImage:
    media_type: str
    width: int
    height: int
    size_bytes: int
    fingerprint: str
    # ไม่แสดงข้อมูลรูปใน repr หรือ log โดยอัตโนมัติ
    data: bytes = field(repr=False)

    def to_claude_block(self) -> dict:
        """เรียกเมื่อจะส่ง API เท่านั้น; ไม่ควรบันทึก block ลง log."""
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": self.media_type,
                "data": base64.b64encode(self.data).decode("ascii"),
            },
        }


def prepare_image(data: bytes) -> PreparedImage:
    """รับ bytes จากไฟล์อัปโหลด และตรวจเนื้อหาจริงแทนนามสกุล."""
    if not isinstance(data, bytes) or not data:
        raise ImageValidationError("กรุณาเลือกไฟล์รูปภาพที่ไม่ว่างค่ะ")
    if len(data) > MAX_IMAGE_BYTES:
        raise ImageValidationError("รูปมีขนาดเกิน 5 MB กรุณาเลือกรูปที่เล็กลงค่ะ")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as picture:
                image_format = picture.format
                if image_format not in SUPPORTED_FORMATS:
                    raise ImageValidationError("รองรับรูป JPG, PNG และ WEBP เท่านั้นค่ะ")
                if getattr(picture, "n_frames", 1) != 1:
                    raise ImageValidationError("กรุณาใช้รูปภาพนิ่งเพียงเฟรมเดียวค่ะ")
                width, height = picture.size
                if (width <= 0 or height <= 0
                        or max(width, height) > MAX_IMAGE_SIDE
                        or width * height > MAX_IMAGE_PIXELS):
                    raise ImageValidationError(
                        "รูปมีความละเอียดสูงเกินไป กรุณาลดขนาดรูปก่อนค่ะ"
                    )
                picture.verify()
            # อ่านข้อมูลพิกเซลด้วย เพื่อจับไฟล์ที่ header ถูกแต่เนื้อหาเสีย
            with Image.open(io.BytesIO(data)) as picture:
                picture.load()
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, EOFError,
            Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ImageValidationError(
            "อ่านรูปไม่ได้หรือไฟล์เสีย กรุณาเลือกรูปใหม่ค่ะ"
        ) from None

    return PreparedImage(
        media_type=SUPPORTED_FORMATS[image_format],
        width=width,
        height=height,
        size_bytes=len(data),
        fingerprint=hashlib.sha256(data).hexdigest(),
        data=data,
    )


def _self_check() -> None:
    # สร้างรูปเล็กในหน่วยความจำ ไม่ต้องใช้รูปของลูกค้า
    for image_format in SUPPORTED_FORMATS:
        buffer = io.BytesIO()
        Image.new("RGB", (32, 24), "white").save(buffer, format=image_format)
        original = buffer.getvalue()
        prepared = prepare_image(original)
        block = prepared.to_claude_block()
        assert base64.b64decode(block["source"]["data"]) == original
        assert (prepared.width, prepared.height) == (32, 24)
        assert prepared.media_type == SUPPORTED_FORMATS[image_format]
        print(f"PASS: {image_format}")

    for label, invalid in (
        ("ไฟล์ว่าง", b""),
        ("ไฟล์ที่ไม่ใช่รูป", b"not an image"),
        ("ไฟล์เกิน 5 MB", b"x" * (MAX_IMAGE_BYTES + 1)),
    ):
        try:
            prepare_image(invalid)
        except ImageValidationError:
            print(f"PASS: ปฏิเสธ{label}")
        else:
            raise AssertionError(f"ไม่ปฏิเสธ{label}")


def main() -> None:
    if len(sys.argv) == 1:
        _self_check()
    elif len(sys.argv) == 2:
        path = Path(sys.argv[1]).expanduser()
        if path.stat().st_size > MAX_IMAGE_BYTES:
            raise ImageValidationError("รูปมีขนาดเกิน 5 MB ค่ะ")
        prepared = prepare_image(path.read_bytes())
        print(f"ชนิดรูป: {prepared.media_type}")
        print(f"ขนาดภาพ: {prepared.width} × {prepared.height} พิกเซล")
        print(f"ขนาดไฟล์: {prepared.size_bytes:,} bytes")
        print("ตรวจรูปสำเร็จ")
    else:
        raise ImageValidationError("ระบุไฟล์รูปได้ครั้งละหนึ่งไฟล์ค่ะ")
    print("ยังไม่ส่งรูปไป Claude และยังไม่เรียก API")


if __name__ == "__main__":
    try:
        main()
    except (ImageValidationError, OSError) as error:
        message = str(error) if isinstance(error, ImageValidationError) else "เปิดไฟล์รูปไม่ได้"
        print(f"ตรวจรูปไม่สำเร็จ: {message}")
        raise SystemExit(1)
