from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

loader = DirectoryLoader("docs/", glob="**/*.txt", loader_cls=TextLoader)
docs = loader.load()

splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedding_model)

llm = Ollama(model="mistral")
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

while True:
    query = input("Ask a question: ")
    if query.lower() in ['exit', 'quit']:
        break
    answer = qa.run(query)
    print("\nAnswer:", answer)