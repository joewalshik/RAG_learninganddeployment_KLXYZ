from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
import os

def load_and_split_text(file_path):
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_documents(documents)

def create_or_load_faiss_index(chunks, embedding_model, index_path="faiss_index"):
    if os.path.exists(index_path):
        print("[INFO] Loading existing FAISS index...")
        return FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
    else:
        print("[INFO] Creating new FAISS index...")
        vectorstore = FAISS.from_documents(chunks, embedding_model)
        vectorstore.save_local(index_path)
        return vectorstore

def run_qa(query, vectorstore, model_name="llama3"):
    llm = Ollama(model=model_name)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True
    )

    result = qa_chain(query)
    print("\n[ANSWER]:", result["result"])

    print("\n[Source Documents]:")
    for doc in result["source_documents"]:
        print("-" * 60)
        print(doc.page_content[:300], '...')
        print("-" * 60)

if __name__ == "__main__":
    file_path = 'documents/books.txt'

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        exit(1)

    chunks = load_and_split_text(file_path)

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = create_or_load_faiss_index(chunks, embedding_model)

    while True:
        user_query = input("\nEnter your question (or type 'exit' to quit):\n> ")
        if user_query.lower() == 'exit':
            break
        run_qa(user_query, vectorstore)