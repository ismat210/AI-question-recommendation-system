# embedding_methods.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class Recommender:
    def __init__(self, questions):
        self.questions = questions
        self.question_texts = [q['question_text'] for q in questions]
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.bert_model = None
        self.bert_embeddings = None

    def fit_tfidf(self):
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.question_texts)
        print("TF-IDF fitted")

    def fit_bert(self, model_name='all-MiniLM-L6-v2'):
        self.bert_model = SentenceTransformer(model_name)
        self.bert_embeddings = self.bert_model.encode(self.question_texts)
        print("BERT embeddings created")

    def recommend_tfidf(self, query, top_k=3):
        query_vec = self.tfidf_vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [self.question_texts[i] for i in top_idx]

    def recommend_bert(self, query, top_k=3):
        query_emb = self.bert_model.encode([query])
        sims = cosine_similarity(query_emb, self.bert_embeddings)[0]
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [self.question_texts[i] for i in top_idx]