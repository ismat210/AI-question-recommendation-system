# evaluate.py
import numpy as np

def top_k_accuracy(recommended_lists, true_indices, k=3):
    hits = 0
    for recs, true_idx in zip(recommended_lists, true_indices):
        if true_idx in recs[:k]:
            hits += 1
    return hits / len(true_indices)

def mean_reciprocal_rank(recommended_lists, true_indices):
    mrr = 0
    for recs, true_idx in zip(recommended_lists, true_indices):
        if true_idx in recs:
            rank = recs.index(true_idx) + 1
            mrr += 1.0 / rank
    return mrr / len(true_indices)