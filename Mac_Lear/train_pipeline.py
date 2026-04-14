import os
import pickle
import numpy as np

# =========================
# CORE IMPORTS (FIXED)
# =========================
from Mac_Lear.data_loader import load_texts
from Mac_Lear.preprocessing import preprocess_list

from Mac_Lear.recommend.tfidf import TFIDFRecommender
from Mac_Lear.recommend.embedding import RecommendationEmbeddingStore
from Mac_Lear.recommend.hybrid import HybridBERTRecommender
from Mac_Lear.topic_tagging import EmbeddingTopicTagger
from Mac_Lear.embedding_generator import EmbeddingGenerator


# =========================
# EVALUATION FUNCTION
# =========================
def evaluate(model, texts):
    scores = []

    for q in texts[:30]:
        try:
            recs = model.search(q, top_k=5)

            rec_texts = [r["question"] for r in recs]

            score = 1 if q in rec_texts else 0
            scores.append(score)

        except Exception:
            continue

    return np.mean(scores) if len(scores) > 0 else 0


# =========================
# TRAIN PIPELINE
# =========================
def train():

    print("\n🚀 STARTING TRAINING PIPELINE\n")

    # 1. LOAD DATA
    texts = load_texts("json")

    # 2. PREPROCESS
    texts = preprocess_list(texts)

    print(f"📊 Total questions: {len(texts)}")

    # 3. TOPIC TAGGING (FIXED)
    tagger = EmbeddingTopicTagger(EmbeddingGenerator())
    tagged = [(t, tagger.predict(t)["topic"]) for t in texts[:5]]

    print("\n📌 Sample Topic Tags:")
    for t in tagged:
        print(t)

    # =========================
    # 4. MODELS
    # =========================
    print("\n🚀 Training Models...\n")

    tfidf = TFIDFRecommender()

    embed = RecommendationEmbeddingStore()
    hybrid = HybridBERTRecommender(alpha=0.7)

    # Fit models
    tfidf.build(texts)
    embed.build(texts)
    hybrid.build(texts)

    print("\n✅ Models trained successfully\n")

    # =========================
    # 5. EVALUATION
    # =========================
    print("📊 Evaluating models...\n")

    scores = {
        "tfidf": evaluate(tfidf, texts),
        "embedding": evaluate(embed, texts),
        "hybrid": evaluate(hybrid, texts),
    }

    for name, score in scores.items():
        print(f"{name.upper()} score: {score:.4f}")

    # =========================
    # 6. BEST MODEL
    # =========================
    best_model_name = max(scores, key=scores.get)

    print("\n🏆 BEST MODEL:", best_model_name.upper())

    model_map = {
        "tfidf": tfidf,
        "embedding": embed,
        "hybrid": hybrid
    }

    best_model = model_map[best_model_name]

    # =========================
    # 7. SAVE MODEL
    # =========================
    os.makedirs("Mac_Lear/models", exist_ok=True)

    with open("Mac_Lear/models/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    print("\n💾 Model saved at: Mac_Lear/models/best_model.pkl")
    print("\n✅ TRAINING COMPLETE\n")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    train()