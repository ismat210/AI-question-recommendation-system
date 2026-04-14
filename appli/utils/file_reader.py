import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from docx import Document
from PIL import Image
import os


# -------------------------
# PDF TEXT EXTRACTION
# -------------------------
def extract_from_pdf(file_path):
    text = ""

    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    except:
        pass

    # If PDF is scanned → fallback OCR
    if len(text.strip()) == 0:
        images = convert_from_path(file_path)
        for img in images:
            text += pytesseract.image_to_string(img)

    return text


# -------------------------
# WORD FILE EXTRACTION
# -------------------------
def extract_from_docx(file_path):
    doc = Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs])


# -------------------------
# IMAGE OCR
# -------------------------
def extract_from_image(file_path):
    img = Image.open(file_path)
    text = pytesseract.image_to_string(img)
    return text


# -------------------------
# MAIN FUNCTION
# -------------------------
def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return extract_from_pdf(file_path)

    elif ext == ".docx":
        return extract_from_docx(file_path)

    elif ext in [".png", ".jpg", ".jpeg"]:
        return extract_from_image(file_path)

    else:
        raise ValueError("Unsupported file type")
if __name__ == "__main__":
    file_path = "data/sample.pdf"

    text = extract_text(file_path)

    print("📄 Extracted Text:\n")
    print(text[:500])  # show first 500 characters