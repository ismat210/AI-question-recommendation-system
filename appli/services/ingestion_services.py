import os

from appli.utils.file_reader import extract_text
from appli.utils.text_cleaning import clean_text
from appli.utils.json_formatter import format_question
from appli.utils.csv_handler import append_to_csv

DATA_PATH = "data/processed/questions.json"


# -------------------------
# LOAD JSON DB
# -------------------------
def load_json():
    if not os.path.exists(DATA_PATH):
        return []

    import json
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# -------------------------
# SAVE JSON DB
# -------------------------
def save_json(data):
    import json

    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


# -------------------------
# MAIN PIPELINE FUNCTION
# -------------------------
def process_file(file_path):
    print("🚀 Starting ingestion pipeline...")

    # 1. Extract raw text
    raw_text = extract_text(file_path)

    # 2. Split into lines (basic question splitting)
    lines = raw_text.split("\n")

    results = []

    # 3. Load existing JSON
    data = load_json()

    for line in lines:
        cleaned = clean_text(line)

        # skip noise
        if len(cleaned) < 5:
            continue

        # 4. Convert to structured JSON
        question_json = format_question(cleaned)

        # 5. Save JSON
        data.append(question_json)

        # 6. Save CSV
        append_to_csv(question_json)

        results.append(question_json)

    # 7. Save JSON file
    save_json(data)

    print(f"✅ Ingestion completed. Total questions: {len(results)}")

    return results


# -------------------------
# TEST RUN
# -------------------------
if __name__ == "__main__":
    file_path = "data/sample.pdf"

    result = process_file(file_path)

    print("\nSample Output:")
    print(result[:2])