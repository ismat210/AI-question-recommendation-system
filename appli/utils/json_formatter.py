import uuid
from datetime import datetime

def format_question(text, topic=None, source="upload"):
    return {
        "id": str(uuid.uuid4()),
        "text": text,
        "topic": topic,
        "source": source,
        "created_at": datetime.utcnow().isoformat()
    }
if __name__ == "__main__":
    sample_text = "Explain linear regression"

    result = format_question(sample_text)

    print(result)