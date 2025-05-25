# app.py
import streamlit as st
from rag_pipeline import build_vector_store
from langchain.llms import Ollama

st.title("Liz-RAG Chatbot")

query = st.text_input("Enter your question:")

if query:
    db = build_vector_store("documents/big.txt")
    docs = db.similarity_search(query, k=3)

    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"Answer the question based on context:\n{context}\nQuestion: {query}"

    llm = Ollama(model="mistral")
    answer = llm(prompt)

    st.write("💬 Answer:")
    st.write(answer)
