import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingTopicTagger:
    """
    Embedding-based topic tagging (no external file dependency)
    """

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

        # ✅ Topics defined inside code (no file dependency = no error)
        self.topics = [
            "Machine Learning",
            "Statistics",
            "Database Systems",
            "Operating Systems",
            "Physics",
            "Biology",
            "Mathematics",
            "Data Structures",
            "Artificial Intelligence",
            "Probability",
            "Linear Algebra",
            "Calculus"
        ]

        # Precompute topic embeddings ONCE (important for speed)
        self.topic_embeddings = self.embedding_model.encode_batch(self.topics)

    # -----------------------------------
    # SINGLE PREDICTION
    # -----------------------------------
    def predict(self, text: str):
        """
        Returns best topic + similarity score
        """

        # Step 1: embed question
        question_embedding = self.embedding_model.encode(text)

        # Step 2: similarity with all topics
        similarities = cosine_similarity(
            [question_embedding],
            self.topic_embeddings
        )[0]

        # Step 3: best match index
        best_idx = int(np.argmax(similarities))

        return {
            "text": text,
            "topic": self.topics[best_idx],
            "score": float(similarities[best_idx])
        }

    # -----------------------------------
    # BATCH PREDICTION
    # -----------------------------------
    def predict_batch(self, texts):
        return [self.predict(t) for t in texts]


# ===================================
# TEST BLOCK
# ===================================
if __name__ == "__main__":
    from Mac_Lear.embedding_generator import EmbeddingGenerator

    # Load embedding model
    embedding_model = EmbeddingGenerator()

    # Initialize tagger
    tagger = EmbeddingTopicTagger(embedding_model)

    # Test questions
    test_questions = [
        "What is linear regression?",
        "Explain SQL join operations",
        "What is photosynthesis?",
        "Define binary tree",
        "What is Bayes theorem?",
        "Explain gradient descent"
    ]

    print("\n🚀 Topic Tagging Results:\n")

    for q in test_questions:
        result = tagger.predict(q)
        print(f"{result['text']} → {result['topic']} ({result['score']:.2f})")