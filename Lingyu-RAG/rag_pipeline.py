from typing import List, Tuple
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Initialize embeddings and vector store
embeddings = OllamaEmbeddings(model="qwen3:8b")
persist_dir = "faiss_db_qwen3_8b"
vectorstore = FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)

# Configure retriever
DESIRED_K = 4
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": DESIRED_K,
        "fetch_k": 5,
        "lambda_mult": 0.5
    }
)

# Define the prompt for condensing questions
condense_question_prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=(
        "Given the following conversation and a follow-up question, "
        "rephrase the follow-up question to be a standalone question.\n\n"
        "Chat History:\n{chat_history}\nFollow-up Question: {question}\nStandalone question:"
    )
)

# Initialize LLM
llm = Ollama(model="qwen3:8b", temperature=0.1)

# Initialize memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# Initialize ConversationalRetrievalChain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    condense_question_prompt=condense_question_prompt,
    return_source_documents=True
)

# Function to remove duplicate documents
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

# Main function to process user queries
def answer_query_modular(query: str) -> Tuple[str, List[Document]]:
    result = qa_chain({"question": query})
    answer_text = result["answer"]
    sources = dedupe_sources(result["source_documents"], k=DESIRED_K)
    return answer_text, sources



