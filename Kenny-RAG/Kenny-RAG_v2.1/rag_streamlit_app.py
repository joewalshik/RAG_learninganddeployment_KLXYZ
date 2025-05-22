import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
import yaml
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

st.set_page_config(page_title="🔍 Local RAG App", layout="centered")
st.title("🔍 Local RAG Chat App with Evaluation")

# Load and split documents
@st.cache_resource
def load_chunks():
    loader = DirectoryLoader("docs/", glob="**/*.txt", loader_cls=TextLoader)
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = []
    for doc in docs:
        for i, chunk in enumerate(splitter.split_documents([doc])):
            chunk.metadata["source"] = doc.metadata.get("source", "local")
            chunk.metadata["page"] = i + 1
            chunks.append(chunk)
    return chunks

chunks = load_chunks()

# Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load or create vector index
if os.path.exists("faiss_index"):
    vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
else:
    vectorstore = FAISS.from_documents(chunks[:50], embedding_model)
    vectorstore.save_local("faiss_index")

# LLM & Prompt
llm = OllamaLLM(model="mistral")
prompt = ChatPromptTemplate.from_template(
    "Answer the question based only on the context below. "
    "If the answer isn't in the context, say 'I don’t know.'\n\n"
    "Context:\n{context}\n\nQuestion:\n{question}"
)
# Sidebar: Temperature control
temperature = st.sidebar.slider("LLM Temperature", 0.0, 1.0, 0.7, 0.1)

# LLM with temperature
llm = OllamaLLM(model="mistral", temperature=temperature)

document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# Query rewriting chain
rewrite_prompt = PromptTemplate.from_template("""
Given the chat history and a follow-up question, rewrite the question to be a standalone question.

Chat history:
{chat_history}

Follow-up question:
{question}

Standalone question:
""")

rewrite_chain = LLMChain(llm=llm, prompt=rewrite_prompt)

# Re-ranking step
def rerank_documents(docs: list[Document], query: str, top_n: int = 5) -> list[Document]:
    def score(doc):
        text = doc.page_content.lower()
        return sum(1 for word in query.lower().split() if word in text)
    return sorted(docs, key=score, reverse=True)[:top_n]

# Evaluation
def evaluate_from_yaml(yaml_path: str):
    with open(yaml_path, "r", encoding="utf-8") as f:
        test_set = yaml.safe_load(f)

    results = []
    for item in test_set:
        question = item["question"]
        expected_keywords = item.get("expected_keywords", [])

        raw_docs = retriever.invoke(question)
        top_docs = rerank_documents(raw_docs, question, top_n=5)
        result = document_chain.invoke({"context": top_docs, "question": question})

        match_count = sum(1 for kw in expected_keywords if kw.lower() in result.lower())
        results.append({
            "question": question,
            "answer": result,
            "match_count": match_count,
            "total_keywords": len(expected_keywords)
        })
    return results

# Streamlit Tabs
tabs = st.tabs(["🔎 Ask", "🧪 Evaluate"])

# Chat Tab
with tabs[0]:
    st.subheader("Chat with Your Local Docs")
    query = st.text_input("Ask a question:", key="user_query")
chat_history = st.session_state.get("chat_history", "")

if query:
    # Rewriting the follow-up question into a standalone one
    rewritten_query = rewrite_chain.run({"chat_history": chat_history, "question": query})

    # Retrieve and generate answer
    raw_docs = retriever.invoke(rewritten_query)
    top_docs = rerank_documents(raw_docs, rewritten_query, top_n=5)
    result = document_chain.invoke({"context": top_docs, "question": rewritten_query})

    # Display results
    st.markdown("**🤖 Answer:**")
    st.markdown(result)

    with st.expander("📄 Sources"):
        for doc in top_docs:
            st.markdown(f"- `{doc.metadata.get('source', '?')}` (Page {doc.metadata.get('page', '-')})")

    # Update chat history
    st.session_state.chat_history = chat_history + f"\nQ: {query}\nA: {result}"


# Evaluation Tab
with tabs[1]:
    st.subheader("Evaluate with Predefined Questions")
    if os.path.exists("test_questions.yaml"):
        results = evaluate_from_yaml("test_questions.yaml")
        for r in results:
            st.markdown(f"**Q:** {r['question']}")
            st.markdown(f"**✅ Matched:** {r['match_count']}/{r['total_keywords']}")
            st.markdown(f"**🧠 Answer:** {r['answer']}")
            st.markdown("---")
    else:
        st.warning("No `test_questions.yaml` file found.")
