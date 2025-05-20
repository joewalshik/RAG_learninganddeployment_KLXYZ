from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
import os

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " "],
    length_function=len
)

documents = []

for filename in os.listdir("documents"):
    if not filename.endswith(".txt"):
        continue

    filepath = os.path.join("documents", filename)
    with open(filepath, "r", encoding="utf-8") as f:
        full_text = f.read()

    lines = full_text.splitlines(keepends=True)

    chunks = text_splitter.split_text(full_text)
    total_chunks = len(chunks)

    offset = 0
    for idx, chunk in enumerate(chunks, start=1):
        preceding_text = full_text[:offset]
        line_number = preceding_text.count("\n") + 1

        metadata = {
            "source": filename,             
            "chunk_index": idx,            
            "total_chunks": total_chunks,   
            "char_start": offset,           
            "line_number": line_number      
        }
        documents.append(Document(page_content=chunk, metadata=metadata))

        offset += len(chunk) - 50  

embeddings = OllamaEmbeddings(model="qwen3:8b")

vectorstore = FAISS.from_documents(
    documents,
    embedding=embeddings
)

persist_dir = "faiss_db_qwen3_8b"
os.makedirs(persist_dir, exist_ok=True)
vectorstore.save_local(persist_dir)

print(f"FAISS index built with line_number metadata and saved to '{persist_dir}'")
