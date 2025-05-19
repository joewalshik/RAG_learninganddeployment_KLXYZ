# 🧠 Local RAG Chat App – Version 2.1

This is an enhanced version of a fully local Retrieval-Augmented Generation (RAG) application built with **Streamlit**, **LangChain**, **FAISS**, and **Ollama**. It introduces flexible LLM temperature control and an intelligent query rewriting mechanism to improve answer relevance and conversational flow — all while keeping your data private and offline.

---

## 🚀 Features

- ✅ **Local-first**: No OpenAI key or internet connection required
- ✅ **Modular LangChain** design (not using `RetrievalQA`)
- ✅ **Custom Prompting**: Forces LLM to only answer using retrieved context or say `"I don’t know"`
- ✅ **User-defined LLM Temperature**: Adjustable creativity/randomness via sidebar slider
- ✅ **Query Rewriting**: Rewrites follow-up questions into standalone queries using chat history
- ✅ **Explicit `top_k` Control** for relevant document retrieval
- ✅ **Chunk Metadata Injection**: Adds source and simulated page number
- ✅ **Simple Reranking Layer**: Keyword-based relevance scoring after vector search
- ✅ **Built-in YAML Evaluation Tool**:
  - Uses `test_questions.yaml` to check how many expected keywords are present
- ✅ **Streamlit UI**:
  - Two tabs: 🔎 Ask questions | 🧪 Evaluate responses
- ✅ **Supports `.txt`, `.pdf`, `.docx` files**
- ✅ **Runs on local models via Ollama** (e.g. `mistral`, `llama2`)
- ✅ **Can be used without uploading documents**

---

## 📁 Project Structure

```
📁 RAG/
├── rag_streamlit_app.py       # Main Streamlit app (Ask + Evaluate tabs)
├── local_rag_app.py           # CLI version of the app
├── test_questions.yaml        # YAML file for evaluation
├── start_app.bat              # Optional script to launch app on Windows
├── requirements.txt           # Dependency list
└── README.md                  # You're reading it!
```

---

## ⚙️ Setup Instructions

### 1. Create and Activate a Virtual Environment

```bash
# Windows
python -m venv venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
# Or manually:
pip install langchain streamlit faiss-cpu langchain-community langchain-ollama langchain-huggingface
```

### 3. Start Ollama

```bash
ollama serve
ollama run mistral
```

(You can replace `mistral` with any supported local model like `llama2`, `gemma`, etc.)

### 4. Launch the App

```bash
streamlit run rag_streamlit_app.py
```

---

## 💬 How to Use

### 🔎 Ask Tab

- Enter a question in the input field
- The app rewrites it (if follow-up) and retrieves top chunks from your local docs
- LLM responds based on the retrieved context
- Adjust LLM temperature via the sidebar to control answer creativity
- Source references are shown (file name + chunk index)

### 🧪 Evaluate Tab

- Place a `test_questions.yaml` in your root directory
- YAML format:
  ```yaml
  - question: "What is a KPI?"
    expected_keywords: ["key performance indicator", "metric"]
  ```
- The app scores each answer by checking how many keywords are matched
- Useful for testing retrieval quality and prompt effectiveness

---

## 🔧 Developer Notes

### ✅ Improvements in This Version

- 🔥 **LLM Temperature**: Control creativity of model outputs
- 🔄 **Query Rewriting**: Improves multi-turn Q&A with context-aware standalone questions

### 🧠 Next Potential Enhancements

- ✍️ **Semantic Reranking** with cross-encoders (e.g. `MiniLM`)
- 💬 **Conversational Memory** via LangChain's memory tools
- 🧪 **Evaluation Metrics** beyond keyword matching (e.g. BLEU, cosine similarity)
- 🔁 **Multi-query expansion** for richer document search

### 🛠 Tech Stack

- `streamlit` · `langchain-core` · `langchain-community`
- `ollama` (local LLM hosting)
- `faiss` (vector similarity search)
- `huggingface` sentence transformers for embedding

---

## 📸 Screenshots 



---

## 🛡 Privacy First

All documents, embeddings, and model inferences are processed **locally on your device**. No internet access or API calls are made. Perfect for confidential or internal data environments.

---

## 🙌 Inspired by

- Azure RAG chat prompt strategies
- LangChain composable workflows
- Ollama offline LLM integrations

---

🚀 Now with temperature + query rewriting. Keep testing and scaling!
