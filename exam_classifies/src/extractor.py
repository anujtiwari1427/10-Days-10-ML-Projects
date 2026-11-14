import fitz  # PyMuPDF
import re
import json

def extract_questions_from_pdf(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    try:
        extracted_items = []
        current_section = "General"
        current_question = ""
        current_q_num = None

        section_pattern = re.compile(r"^(SECTION\s*[-–:]?\s*[A-Z0-9]+)", re.IGNORECASE)
        question_start_pattern = re.compile(r"^(?:Q\.?\s*)?(\d+)[\.\)]\s+(.*)")

        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("blocks")
            page_height = page.rect.height

            for b in blocks:
                x0, y0, x1, y1, text, block_no, block_type = b
                if block_type != 0 or y0 < page_height * 0.07 or y1 > page_height * 0.93:
                    continue

                for line in text.strip().split("\n"):
                    line = line.strip()
                    if not line:
                        continue

                    sec_match = section_pattern.match(line)
                    if sec_match:
                        current_section = sec_match.group(1).upper()
                        continue

                    q_match = question_start_pattern.match(line)
                    if q_match:
                        if current_question:
                            extracted_items.append({
                                "q_num": current_q_num,
                                "section": current_section,
                                "page": page_num + 1,
                                "text": current_question.strip()
                            })
                        current_q_num = q_match.group(1)
                        current_question = q_match.group(2)
                    else:
                        if current_question:
                            current_question += " " + line

        if current_question:
            extracted_items.append({
                "q_num": current_q_num,
                "section": current_section,
                "page": len(doc),
                "text": current_question.strip()
            })

        return extracted_items
    finally:
        doc.close()