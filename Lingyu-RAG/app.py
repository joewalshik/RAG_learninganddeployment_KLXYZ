import streamlit as st
from rag_pipeline import answer_query_modular

st.title("Local RAG Chatbot with Source Attribution and Metadata")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Enter your question:")

if query:
    answer, sources = answer_query_modular(query)
    st.session_state.chat_history.append(("User", query))
    st.session_state.chat_history.append(("Assistant", answer))

    st.write("**Answer:**")
    st.write(answer)

    st.write("**Sources (with full metadata):**")
    for i, doc in enumerate(sources):
        md = doc.metadata
        filename = md.get("source", "Unknown")
        chunk_idx = md.get("chunk_index", "Unknown")
        total_chunks = md.get("total_chunks", "Unknown")
        char_start = md.get("char_start", "Unknown")
        line_number = md.get("line_number", "Unknown")

        st.write(f"**Source {i+1}:**")
        st.write(f"- File: {filename}")
        st.write(f"- Chunk Index: {chunk_idx} / {total_chunks}")
        st.write(f"- Starting Character: {char_start}")
        st.write(f"- Starting Line: {line_number}")
        st.write("```")
        st.write(doc.page_content)
        st.write("```")
        st.write("---")

