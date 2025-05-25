import os
import requests
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Automatically download big.txt (if it does not exist)
BIGTXT_URL = "https://norvig.com/big.txt"
LOCAL_FILE = "big.txt"
if not os.path.exists(LOCAL_FILE):
    print("big.txt does not exist, downloading...")
    r = requests.get(BIGTXT_URL)
    with open(LOCAL_FILE, "wb") as f:
        f.write(r.content)
    print("Download complete!")

# Load big.txt
with open(LOCAL_FILE, encoding="utf-8") as f:
    raw_text = f.read()

# Split the text into paragraphs
documents = [p for p in raw_text.split('\n') if p.strip()]

# Create TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(documents)

app = Flask(__name__)

def ask_ollama(model, prompt):
    url = "http://localhost:11434/api/generate"
    response = requests.post(
        url,
        json={
            "model": model,         
            "prompt": prompt,
            "stream": False         
        }
    )
    data = response.json()
    return data.get("response", "").strip()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    query = request.json.get("query", "")
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    idx = similarities.argmax()
    answer = documents[idx]
    context = documents[idx]

    # Construct prompt for Ollama
    prompt = (
    f"Based on the following information:\n\n{context}\n\n"
    f"Please answer the user's question according to the above content: {query}\n"
    f"If the information above cannot answer the question, please simply say so."
    )
    answer = ask_ollama("llama2", prompt)  

    return jsonify({"result": answer})

if __name__ == "__main__":
    app.run(debug=True)