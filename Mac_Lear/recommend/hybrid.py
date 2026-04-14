import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class HybridBERTRecommender:
    """
    Hybrid Recommender:
    TF-IDF (lexical) + BERT (semantic)
    """

    def __init__(self, alpha=0.7, model_name="all-MiniLM-L6-v2"):
        """
        alpha = weight for BERT
        (1 - alpha) = weight for TF-IDF
        """

        self.alpha = alpha

        # TF-IDF model
        self.tfidf = TfidfVectorizer(stop_words="english")

        # BERT model
        self.model = SentenceTransformer(model_name)

        self.questions = []
        self.tfidf_matrix = None
        self.embeddings = None

    # -----------------------------------
    # BUILD MODEL
    # -----------------------------------
    def build(self, questions):
        """
        Fit TF-IDF + BERT embeddings
        """

        self.questions = questions

        # TF-IDF matrix
        self.tfidf_matrix = self.tfidf.fit_transform(questions)

        # BERT embeddings
        self.embeddings = self.model.encode(questions, convert_to_numpy=True)

        print(f"✅ Hybrid model built on {len(questions)} questions")

    # -----------------------------------
    # ADD NEW QUESTION
    # -----------------------------------
    def add_question(self, question):
        """
        Online update (simple retrain TF-IDF + append embedding)
        """

        self.questions.append(question)

        # retrain TF-IDF (safe approach)
        self.tfidf_matrix = self.tfidf.fit_transform(self.questions)

        # add BERT embedding
        emb = self.model.encode(question)

        if self.embeddings is None:
            self.embeddings = np.array([emb])
        else:
            self.embeddings = np.vstack([self.embeddings, emb])

    # -----------------------------------
    # SEARCH FUNCTION
    # -----------------------------------
    def search(self, query, top_k=5):
        """
        Hybrid ranking system
        """

        if len(self.questions) == 0:
            return []

        # -------------------------
        # TF-IDF similarity
        # -------------------------
        query_tfidf = self.tfidf.transform([query])
        tfidf_scores = cosine_similarity(query_tfidf, self.tfidf_matrix)[0]

        # -------------------------
        # BERT similarity
        # -------------------------
        query_emb = self.model.encode(query)

        bert_scores = np.array([
            np.dot(query_emb, doc_emb) /
            (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
            for doc_emb in self.embeddings
        ])

        # -------------------------
        # NORMALIZATION (NumPy 2.0 FIXED)
        # -------------------------
        tfidf_scores = (
            (tfidf_scores - tfidf_scores.min()) /
            (np.ptp(tfidf_scores) + 1e-8)
        )

        bert_scores = (
            (bert_scores - bert_scores.min()) /
            (np.ptp(bert_scores) + 1e-8)
        )

        # -------------------------
        # FINAL HYBRID SCORE
        # -------------------------
        final_scores = self.alpha * bert_scores + (1 - self.alpha) * tfidf_scores

        # -------------------------
        # TOP-K SELECTION
        # -------------------------
        top_idx = final_scores.argsort()[-top_k:][::-1]

        results = [
            {
                "question": self.questions[i],
                "score": float(final_scores[i]),
                "bert_score": float(bert_scores[i]),
                "tfidf_score": float(tfidf_scores[i])
            }
            for i in top_idx
        ]

        return results


# ===================================
# TEST BLOCK
# ===================================
if __name__ == "__main__":

    recommender = HybridBERTRecommender(alpha=0.7)

    questions = [
        "What is linear regression?",
        "Explain logistic regression",
        "What is SQL join?",
        "Define binary tree",
        "What is photosynthesis?",
        "Explain gradient descent",
        "What is overfitting in machine learning?",
        "What is hypothesis testing?"
    ]

    # Build system
    recommender.build(questions)

    # Query
    query = "Explain regression in machine learning"

    results = recommender.search(query, top_k=3)

    print("\n🚀 Hybrid BERT Recommendations:\n")

    for r in results:
        print(
            r["question"],
            "→ score:", round(r["score"], 3),
            "| bert:", round(r["bert_score"], 3),
            "| tfidf:", round(r["tfidf_score"], 3)
        )