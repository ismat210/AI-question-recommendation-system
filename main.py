# main.py
from utils import load_questions
from em_methods import Recommender
from evaluate import top_k_accuracy, mean_reciprocal_rank
import random

# Load questions
questions = load_questions()

# Initialize Recommender
rec_sys = Recommender(questions)

# Fit embeddings
rec_sys.fit_tfidf()
rec_sys.fit_bert()

# Simulate queries
num_queries = max(1, len(questions)//5)
query_indices = random.sample(range(len(questions)), num_queries)
queries = [questions[i]['question_text'] for i in query_indices]

# Evaluate TF-IDF
tfidf_recs = [rec_sys.recommend_tfidf(q, top_k=3) for q in queries]
# Map recommended texts back to indices
tfidf_rec_indices = [[rec_sys.question_texts.index(r) for r in recs] for recs in tfidf_recs]
tfidf_top1 = top_k_accuracy(tfidf_rec_indices, query_indices, k=1)
tfidf_mrr = mean_reciprocal_rank(tfidf_rec_indices, query_indices)

# Evaluate BERT
bert_recs = [rec_sys.recommend_bert(q, top_k=3) for q in queries]
bert_rec_indices = [[rec_sys.question_texts.index(r) for r in recs] for recs in bert_recs]
bert_top1 = top_k_accuracy(bert_rec_indices, query_indices, k=1)
bert_mrr = mean_reciprocal_rank(bert_rec_indices, query_indices)

print(f"TF-IDF Top-1 Accuracy: {tfidf_top1:.2f}, MRR: {tfidf_mrr:.2f}")
print(f"BERT Top-1 Accuracy: {bert_top1:.2f}, MRR: {bert_mrr:.2f}")