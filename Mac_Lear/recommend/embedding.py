import numpy as np
from Mac_Lear.embedding_generator import EmbeddingGenerator


class RecommendationEmbeddingStore:
    """
    Stores embeddings for questions and supports similarity search
    """

    def __init__(self):
        self.embedder = EmbeddingGenerator()

        # storage
        self.questions = []
        self.embeddings = None

    # -----------------------------------
    # BUILD EMBEDDING DATABASE
    # -----------------------------------
    def build(self, questions):
        """
        questions: list of text questions
        """

        self.questions = questions
        self.embeddings = self.embedder.encode_batch(questions)

        print(f"✅ Built embedding store for {len(questions)} questions")

    # -----------------------------------
    # ADD NEW QUESTION (ONLINE LEARNING)
    # -----------------------------------
    def add_question(self, question: str):
        """
        Add single new question dynamically
        """

        emb = self.embedder.encode(question)

        self.questions.append(question)

        if self.embeddings is None:
            self.embeddings = np.array([emb])
        else:
            self.embeddings = np.vstack([self.embeddings, emb])

    # -----------------------------------
    # SIMILARITY SEARCH
    # -----------------------------------
    def search(self, query: str, top_k=5):
        """
        Return top-K similar questions
        """

        if len(self.questions) == 0:
            return []

        query_emb = self.embedder.encode(query)

        scores = []

        for emb in self.embeddings:
            score = self.embedder.similarity(query_emb, emb)
            scores.append(score)

        scores = np.array(scores)

        # top-k indices
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

    store = RecommendationEmbeddingStore()

    questions = [
        "What is linear regression?",
        "Explain logistic regression",
        "What is SQL join?",
        "Define binary tree",
        "What is photosynthesis?",
        "Explain gradient descent"
    ]

    # Build index
    store.build(questions)

    # Query test
    query = "Explain regression in ML"

    results = store.search(query, top_k=3)

    print("\n🔍 Top Recommendations:\n")

    for r in results:
        print(r["question"], "→", round(r["score"], 3))