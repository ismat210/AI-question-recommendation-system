import csv
import os

CSV_PATH = "data/processed/questions.csv"

def append_to_csv(question_json):
    # ✅ Create folder if not exists
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

    file_exists = os.path.isfile(CSV_PATH)

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=question_json.keys())

        # Write header only once
        if not file_exists:
            writer.writeheader()

        writer.writerow(question_json)
if __name__ == "__main__":
    sample = {
        "id": "1",
        "text": "What is regression?",
        "topic": "Regression",
        "source": "test",
        "created_at": "2026-04-14"
    }

    append_to_csv(sample)
    print("✅ Added to CSV")