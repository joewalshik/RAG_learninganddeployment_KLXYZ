from typing import List
import os

from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.chains import StuffDocumentsChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document

embeddings = OllamaEmbeddings(model="qwen3:8b") 
persist_dir = "faiss_db_qwen3_8b"               
vectorstore = FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True) 

DESIRED_K = 4
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": DESIRED_K,     
        "fetch_k": 5,       
        "lambda_mult": 0.5 
    }
)

document_prompt = PromptTemplate(
    input_variables=["page_content", "source", "chunk_index", "total_chunks", "line_number"],
    template=(
        "Chunk {chunk_index}/{total_chunks} from {source} (Line {line_number}):\n"
        "{page_content}\n"
        "-----\n"
    )
)

llm_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are an AI assistant text reader. Use the following retrieved context to answer the question.\n\n"
        "CONTEXT:\n"
        "{context}\n\n"
        "QUESTION:\n"
        "{question}\n\n"
        "ANSWER:"
    )
)

llm = Ollama(
    model="qwen3:8b",
    temperature=0.1
)

llm_chain = LLMChain(
    llm=llm,
    prompt=llm_prompt,
    output_key="text"
)

combine_documents_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_variable_name="context",
    document_prompt=document_prompt,
    output_key="answer"
)

def dedupe_sources(source_docs: List[Document], k: int = DESIRED_K) -> List[Document]:

    seen = set()
    unique = []
    for doc in source_docs:
        content = doc.page_content.strip()
        if content not in seen:
            unique.append(doc)
            seen.add(content)
        if len(unique) >= k:
            break
    return unique 

def answer_query_modular(query: str):
    raw_docs: List[Document] = retriever.get_relevant_documents(query)
    result = combine_documents_chain({
        "input_documents": raw_docs,
        "question": query
    })

    answer_text = result["answer"] 
    sources = dedupe_sources(raw_docs, k=DESIRED_K) 

    return answer_text, sources



