import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# -----------------------------
# CLEAN TEXT FOR ML
# -----------------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9?.!, ]", "", text)
    return text.strip()


# -----------------------------
# REMOVE STOPWORDS (ML VERSION)
# -----------------------------
def remove_stopwords(text: str) -> str:
    words = text.split()
    words = [w for w in words if w not in ENGLISH_STOP_WORDS]
    return " ".join(words)


# -----------------------------
# FULL PREPROCESS PIPELINE
# -----------------------------
def preprocess_text(text: str) -> str:
    text = clean_text(text)
    text = remove_stopwords(text)
    return text


# -----------------------------
# LIST PREPROCESSING
# -----------------------------
def preprocess_list(texts):
    return [preprocess_text(t) for t in texts]


# -----------------------------
# TEST BLOCK
# -----------------------------
if __name__ == "__main__":
    sample = [
        "What is Linear Regression???",
        "Explain   classification in ML!!!",
        "Define overfitting in machine learning."
    ]

    processed = preprocess_list(sample)

    print("Original:", sample)
    print("Processed:", processed)