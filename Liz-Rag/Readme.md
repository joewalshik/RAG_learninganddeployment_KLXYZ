# Liz-RAG Chatbot

Liz-RAG is a fully local Retrieval-Augmented Generation (RAG) chatbot that allows users to ask natural language questions based on custom text documents. It uses FAISS for semantic search, HuggingFace embeddings for vector representation, and the Mistral language model running locally via Ollama for answer generation. The system is entirely offline and provides a web interface via Streamlit.

<img width="673" alt="image" src="https://github.com/user-attachments/assets/cd58ab24-c266-49e0-8554-689d5487d850" />


## Features

The application supports custom `.txt` documents, splits them into text chunks for embedding using sentence-transformers, stores the resulting vectors using FAISS, and uses a locally hosted Mistral model to answer user questions. The entire pipeline is orchestrated using LangChain, and the frontend is implemented with Streamlit to provide an interactive user interface.

## Project Structure

The project consists of the following main files and folders: `app.py` handles the Streamlit frontend; `rag_pipeline.py` contains the logic for document processing, embedding, and vector search; the `documents/` folder holds input files such as `big.txt`; and `requirements.txt` lists all Python dependencies. The virtual environment (`rag_venv/`) is used for isolation but excluded from version control.

## Getting Started

To run this project, first clone the repository and enter the project directory:

```
git clone <your-repo-url>
cd Liz-RAG
```

Then create a virtual environment and activate it:

```
python3 -m venv rag_venv
source rag_venv/bin/activate
```

Next, install the required Python packages:

```
pip install --upgrade pip
pip install -r requirements.txt
```

If you don’t have a `requirements.txt` file, you can install dependencies manually:

```
pip install langchain langchain-community streamlit sentence-transformers faiss-cpu transformers
```

Create the `documents` directory if it doesn't exist, and download a sample document:

```
mkdir -p documents
curl -o documents/big.txt https://norvig.com/big.txt
```

Ensure you have [Ollama](https://ollama.com/) installed. Then pull and serve the model:

```
ollama pull mistral
ollama serve
```

Once the model is ready, launch the chatbot application:

```
streamlit run app.py
```

Open your browser and navigate to `http://localhost:8501` to interact with the chatbot.

## Example Usage

Once the interface loads, you can enter questions like:

```
What is the most common word in the document?
```

The application will perform document retrieval and generate an answer using the local LLM.

## Acknowledgements

This project uses open-source tools including LangChain for orchestration, HuggingFace for embeddings, FAISS for vector storage, and Ollama for local language model inference. The sample document `big.txt` is sourced from Peter Norvig’s public corpus.

