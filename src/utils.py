# utils.py
import json

def load_questions(json_path="questions.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_question_texts(questions):
    return [q['question_text'] for q in questions]