import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class BERTHybridRecommender:
    """
    BERT-based semantic recommender system
    (replaces TF-IDF with deep embeddings)
    """

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        # BERT / Transformer model
        self.model = SentenceTransformer(model_name)

        self.questions = []
        self.embeddings = None

    # -----------------------------------
    # BUILD INDEX
    # -----------------------------------
    def build(self, questions):
        """
        Encode all questions into BERT embeddings
        """

        self.questions = questions
        self.embeddings = self.model.encode(questions, convert_to_numpy=True)

        print(f"✅ BERT index built for {len(questions)} questions")

    # -----------------------------------
    # ADD NEW QUESTION (ONLINE UPDATE)
    # -----------------------------------
    def add_question(self, question):
        """
        Add new question dynamically
        """

        self.questions.append(question)

        emb = self.model.encode(question)

        if self.embeddings is None:
            self.embeddings = np.array([emb])
        else:
            self.embeddings = np.vstack([self.embeddings, emb])

    # -----------------------------------
    # SEARCH
    # -----------------------------------
    def search(self, query, top_k=5):
        """
        Return top-K semantically similar questions
        """

        if len(self.questions) == 0:
            return []

        query_emb = self.model.encode(query)

        scores = cosine_similarity(
            [query_emb],
            self.embeddings
        )[0]

        top_idx = scores.argsort()[-top_k:][::-1]

        results = [
            {
                "question": self.questions[i],
                "score": float(scores[i])
            }
            for i in top_idx
        ]

        return results


# ===================================
# TEST BLOCK
# ===================================
if __name__ == "__main__":

    recommender = BERTHybridRecommender()

    questions = [
        "What is linear regression?",
        "Explain logistic regression",
        "What is SQL join?",
        "Define binary tree",
        "What is photosynthesis?",
        "Explain gradient descent",
        "What is overfitting?"
    ]

    recommender.build(questions)

    query = "Explain regression in machine learning"

    results = recommender.search(query, top_k=3)

    print("\n🚀 BERT Recommendations:\n")

    for r in results:
        print(r["question"], "→", round(r["score"], 3))