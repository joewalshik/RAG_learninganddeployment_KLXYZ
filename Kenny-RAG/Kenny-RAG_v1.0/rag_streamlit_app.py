import os
import tempfile
import streamlit as st
from datetime import datetime
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# App setup
st.set_page_config(page_title="🧠 NCL RAG Chat", layout="centered")
st.title("🧠 NCL RAG Chat")

# State defaults
st.session_state.setdefault("qa_chain", None)
st.session_state.setdefault("messages", [])

# Upload files
uploaded_files = st.file_uploader(
    "Upload your TXT / PDF / DOCX files", 
    accept_multiple_files=True, 
    type=["txt", "pdf", "docx"]
)

# Helper: Load & Split files
@st.cache_resource
def load_documents_and_vectorstore(uploaded_files):
    temp_dir = tempfile.mkdtemp()
    all_chunks = []

    for file in uploaded_files:
        file_path = os.path.join(temp_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())

        if file.name.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
        elif file.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file.name.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            continue

        docs = loader.load()
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        all_chunks.extend(chunks)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(all_chunks, embeddings)
    return vectorstore

# Load pipeline if files uploaded
if uploaded_files:
    with st.spinner("Processing documents..."):
        vectorstore = load_documents_and_vectorstore(uploaded_files)
        retriever = vectorstore.as_retriever()
        llm = Ollama(model="mistral")
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_query = st.chat_input("Ask a question...")

if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    if not user_query.strip():
        with st.chat_message("assistant"):
            st.warning("Please enter a valid question.")
        st.session_state.messages.append({"role": "assistant", "content": "Please enter a valid question."})
    elif st.session_state.qa_chain is None:
        with st.chat_message("assistant"):
            st.warning("No documents have been uploaded yet. Please upload a file to enable document-based answers.")
        st.session_state.messages.append({"role": "assistant", "content": "Please upload a file to enable document-based answers."})
    else:
        with st.spinner("Thinking..."):
            result = st.session_state.qa_chain(user_query)
            answer = result['answer']
            sources = result.get("sources", "")
            response_text = answer + (f"\n\n📄 **Sources:** {sources}" if sources else "")

        with st.chat_message("assistant"):
            st.markdown(answer)
            if sources:
                st.markdown(f"📄 **Sources:** {sources}")

        st.session_state.messages.append({"role": "assistant", "content": answer})

st.markdown("---")
st.caption("🔒 All conversations are kept local and private.")
