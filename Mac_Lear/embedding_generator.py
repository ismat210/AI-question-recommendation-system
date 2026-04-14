import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    """
    Converts text → embeddings for recommendation system
    """

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        # Load pretrained transformer model
        self.model = SentenceTransformer(model_name)

    # -----------------------------------
    # SINGLE TEXT EMBEDDING
    # -----------------------------------
    def encode(self, text: str):
        return self.model.encode(text)

    # -----------------------------------
    # BATCH ENCODING (IMPORTANT FOR SPEED)
    # -----------------------------------
    def encode_batch(self, texts):
        return self.model.encode(texts, convert_to_numpy=True)

    # -----------------------------------
    # NORMALIZED EMBEDDING (BETTER FOR SIMILARITY)
    # -----------------------------------
    def encode_normalized(self, texts):
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms

    # -----------------------------------
    # COSINE SIMILARITY
    # -----------------------------------
    def similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (
            np.linalg.norm(vec1) * np.linalg.norm(vec2)
        )


# ===================================
# TEST BLOCK
# ===================================
if __name__ == "__main__":
    embedding_model = EmbeddingGenerator()

    questions = [
        "What is linear regression?",
        "Explain regression in machine learning",
        "What is photosynthesis?",
        "Define binary tree",
        "What is SQL join?"
    ]

    print("\n🚀 Embedding Generator Test\n")

    # Step 1: batch embeddings
    embeddings = embedding_model.encode_batch(questions)

    print("Embedding shape:", embeddings.shape)

    # Step 2: similarity test
    sim = embedding_model.similarity(embeddings[0], embeddings[1])

    print("\nSimilarity Example:")
    print(f"Q1: {questions[0]}")
    print(f"Q2: {questions[1]}")
    print("Similarity:", round(sim, 3))