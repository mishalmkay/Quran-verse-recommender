import pickle
import boto3
import os
import re
import nltk
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.metrics.pairwise import cosine_similarity

# Ensure necessary NLTK downloads
nltk.download('punkt')
nltk.download('punkt_tab')

nltk.download('stopwords')

# AWS S3 Config
BUCKET_NAME = "recommender-system-32"
LOCAL_PATH = "./data/"
os.makedirs(LOCAL_PATH, exist_ok=True)

# S3 Client
s3 = boto3.client("s3")

def download_from_s3(file_name):
    local_file_path = os.path.join(LOCAL_PATH, file_name)
    s3.download_file(BUCKET_NAME, file_name, local_file_path)
    return local_file_path

# Download pickled files
quran_df_path = download_from_s3("quran_df.pkl")
vectorizer_path = download_from_s3("vectorizer.pkl")
tfidf_matrix_path = download_from_s3("tfidf_matrix.pkl")

# Load Pickled Data
with open(quran_df_path, "rb") as file:
    quran_df = pickle.load(file)

with open(vectorizer_path, "rb") as file:
    vectorizer = pickle.load(file)

with open(tfidf_matrix_path, "rb") as file:
    tfidf_matrix = pickle.load(file)

# Flask App
app = Flask(_name_)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    words = nltk.word_tokenize(text)
    stop_words = set(nltk.corpus.stopwords.words("english"))
    words = [word for word in words if word not in stop_words]

    stemmer = nltk.stem.PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    return " ".join(words)

# Home Page
@app.route("/")
def home():
    return render_template("index.html")

# Recommendation API
@app.route("/recommend", methods=["POST"])
def get_recommendations():
    query = request.form.get("query")
    top_n = int(request.form.get("top_n", 5))

    if not query:
        return jsonify({"error": "Query is required"}), 400

    processed_query = preprocess_text(query)
    query_vector = vectorizer.transform([processed_query])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    recommendations = quran_df.iloc[top_indices][["surah", "verse", "text"]].copy()
    recommendations["similarity"] = similarity_scores[top_indices]

    return render_template("results.html", results=recommendations.to_dict(orient="records"))

if _name_ == "_main_":
    app.run(host="0.0.0.0", port=5000, debug=True)
