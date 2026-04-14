import json
import pandas as pd
import os


# -------------------------
# LOAD FROM JSON
# -------------------------
def load_json_data(path="data/processed/questions.json"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


# -------------------------
# LOAD FROM CSV
# -------------------------
def load_csv_data(path="data/processed/questions.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)
    return df


# -------------------------
# LOAD TEXTS FOR ML
# -------------------------
def load_texts(source="json"):
    """
    Returns list of questions for ML models
    """

    if source == "json":
        data = load_json_data()
        texts = [item["text"] for item in data]

    elif source == "csv":
        df = load_csv_data()
        texts = df["text"].tolist()

    else:
        raise ValueError("source must be 'json' or 'csv'")

    return texts
if __name__ == "__main__":
    texts = load_texts("json")

    print("Total questions:", len(texts))
