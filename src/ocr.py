# ocr_and_clean.py
import easyocr
import re
import json
from PIL import Image
import numpy as np
from pdf2image import convert_from_path

def extract_text_from_pdf(pdf_path):
    """
    Convert PDF pages to images and extract text using EasyOCR
    """
    reader = easyocr.Reader(['en'])
    pages = convert_from_path(pdf_path)
    all_text = ""

    for i, page in enumerate(pages):
        result = reader.readtext(np.array(page), detail=0)
        page_text = " ".join(result)
        all_text += page_text + "\n"
    return all_text

def parse_questions(text):
    """
    Split OCR text into questions and clean header/footer
    """
    questions = []
    # Split by "Problem" or numbering
    raw_questions = re.split(r'Problem\s+[IVX]+[:.]|\n\d+\.', text, flags=re.IGNORECASE)

    for q in raw_questions:
        q = q.strip()
        if not q or len(q) < 30:  # ignore very short lines
            continue
        # Remove header/footer text
        if "assignment" in q.lower() or "instructor" in q.lower():
            continue
        questions.append({
            "question_text": q,
            "options": [],
            "correct_answer": None
        })
    return questions

def save_questions_to_json(questions, json_path="questions.json"):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)

# Example usage
if __name__ == "__main__":
    pdf_path = "sample_questions.pdf"
    text = extract_text_from_pdf(pdf_path)
    questions = parse_questions(text)
    save_questions_to_json(questions)
    print(f"Saved {len(questions)} questions to questions.json")