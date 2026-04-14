import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFRecommender:
    """
    TF-IDF based recommendation system (baseline model)
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.questions = []
        self.tfidf_matrix = None

    # -----------------------------------
    # BUILD MODEL
    # -----------------------------------
    def build(self, questions):
        """
        Fit TF-IDF on question corpus
        """

        self.questions = questions
        self.tfidf_matrix = self.vectorizer.fit_transform(questions)

        print(f"✅ TF-IDF model built on {len(questions)} questions")

    # -----------------------------------
    # ADD NEW QUESTION (ONLINE UPDATE)
    # -----------------------------------
    def add_question(self, question):
        """
        Add new question (simple retrain for consistency)
        """

        self.questions.append(question)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.questions)

    # -----------------------------------
    # SEARCH SIMILAR QUESTIONS
    # -----------------------------------
    def search(self, query, top_k=5):
        """
        Return top-K similar questions
        """

        if len(self.questions) == 0:
            return []

        query_vec = self.vectorizer.transform([query])

        scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]

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

    recommender = TFIDFRecommender()

    questions = [
        "What is linear regression?",
        "Explain logistic regression",
        "What is SQL join?",
        "Define binary tree",
        "What is photosynthesis?",
        "Explain gradient descent"
    ]

    # Build model
    recommender.build(questions)

    # Query
    query = "Explain regression in machine learning"

    results = recommender.search(query, top_k=3)

    print("\n🔍 TF-IDF Recommendations:\n")

    for r in results:
        print(r["question"], "→", round(r["score"], 3))