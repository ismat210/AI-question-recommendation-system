import re

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9?.!, ]", "", text)
    return text.strip()
if __name__ == "__main__":
    sample = "   EXPLAIN Regression!!!   "
    
    cleaned = clean_text(sample)
    
    print("Original:", sample)
    print("Cleaned:", cleaned)