# 🧠 Local RAG Chat App

This is a lightweight, local Retrieval-Augmented Generation (RAG) application built using Streamlit, LangChain, FAISS, and Ollama.

It allows users to upload `.txt`, `.pdf`, or `.docx` documents and ask natural language questions based on their contents — **without relying on external APIs or cloud services**.

---

## 🚀 Features

- ✅ Local-only: No internet connection or OpenAI key required
- ✅ Supports TXT, PDF, and Word documents
- ✅ Powered by Ollama (e.g. Mistral or LLaMA2 running locally)
- ✅ Conversational memory — follow-up questions are supported
- ✅ Easy to use Streamlit interface
- ✅ Document upload optional — app still accepts general questions

---

## 📂 Project Structure

📁 RAG/
├── rag_streamlit_app.py # Main Streamlit app
├── requirements.txt # Python dependencies
└── README.md # You're reading it!


---
## 🧪 Setup Instructions

# Install Ollama and pull a model

ollama run mistral 
(You can also use llama2, gemma, or any other supported local model.)


# **Create a virtual environment** (recommended):


python -m venv venv

on window:
.\venv\Scripts\activate 

On macOS/Linux:
source venv/bin/activate

# Install dependencies

pip install -r requirements.txt

# Run the App
streamlit run rag_streamlit_app.py

# How to Use

Upload one or more .txt, .pdf, or .docx files

Ask any question related to your content

If no documents are uploaded, the assistant will still respond — but document-specific answers won’t be available

All data is processed locally and never saved

## Privacy Note

No data is sent to external APIs or services. This tool runs entirely offline and is safe for use with confidential or internal documents.

---
