# app.py
from flask import Flask, request, jsonify
from utils import load_questions
from embedding_methods import Recommender

app = Flask(__name__)

# Load questions and fit embeddings
questions = load_questions()
rec_sys = Recommender(questions)
rec_sys.fit_tfidf()
rec_sys.fit_bert()

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    query = data.get("query", "")
    method = data.get("method", "bert")  # "bert" or "tfidf"
    top_k = int(data.get("top_k", 3))

    if method == "tfidf":
        recs = rec_sys.recommend_tfidf(query, top_k=top_k)
    else:
        recs = rec_sys.recommend_bert(query, top_k=top_k)

    return jsonify({"query": query, "recommendations": recs})

if __name__ == "__main__":
    app.run(debug=True)