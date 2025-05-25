from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document
import os
import yaml

# Load and split documents
loader = DirectoryLoader("docs/", glob="**/*.txt", loader_cls=TextLoader)
docs = loader.load()

splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = []
for doc in docs:
    for i, chunk in enumerate(splitter.split_documents([doc])):
        chunk.metadata["source"] = doc.metadata.get("source", "local")
        chunk.metadata["page"] = i + 1
        chunks.append(chunk)

# Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# === Load or create vector index ===
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

document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# Re-ranking step
def rerank_documents(docs: list[Document], query: str, top_n: int = 5) -> list[Document]:
    def score(doc):
        text = doc.page_content.lower()
        return sum(1 for word in query.lower().split() if word in text)
    return sorted(docs, key=score, reverse=True)[:top_n]

# Evaluation Mode
def evaluate_from_yaml(yaml_path: str):
    with open(yaml_path, "r", encoding="utf-8") as f:
        test_set = yaml.safe_load(f)

    print("\n🧪 Evaluation Results:")
    for item in test_set:
        question = item["question"]
        expected_keywords = item.get("expected_keywords", [])

        raw_docs = retriever.invoke(question)
        top_docs = rerank_documents(raw_docs, question, top_n=5)
        result = document_chain.invoke({"context": top_docs, "question": question})

        match_count = sum(1 for kw in expected_keywords if kw.lower() in result.lower())
        print(f"\nQ: {question}\n→ ✅ Matched {match_count}/{len(expected_keywords)} expected keywords")
        print(f"→ Answer: {result}")

# CLI Chat Loop
if os.path.exists("test_questions.yaml"):
    evaluate_from_yaml("test_questions.yaml")
else:
    print("💬 RAG Chat (type 'exit' to quit)")
    while True:
        query = input("\nAsk a question: ")
        if query.lower() in ["exit", "quit"]:
            break

        raw_docs = retriever.invoke(query)
        top_docs = rerank_documents(raw_docs, query, top_n=5)
        result = document_chain.invoke({"context": top_docs, "question": query})

        print("\n🤖 Answer:", result)
        print("\n📄 Sources:")
        for doc in top_docs:
            print(f"- {doc.metadata.get('source', '?')} (Page {doc.metadata.get('page', '-')})")
