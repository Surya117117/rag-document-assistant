# 📄 RAG Document Assistant

A modular, traditional **Retrieval-Augmented Generation (RAG)** pipeline that lets you ask natural language questions about your own documents. Built with LangChain, ChromaDB, Google Gemini, and Sentence Transformers — designed with a clean, folder-based architecture so every component is easy to understand, swap, or extend.



## 🧠 What is RAG?

RAG (Retrieval-Augmented Generation) is a technique that combines a **document retrieval system** with a **large language model (LLM)**. Instead of relying solely on what the LLM was trained on, RAG first fetches the most relevant chunks from your own documents, then feeds them as context to the LLM to generate a grounded, accurate answer.



Your Question
     │
     ▼
[Retriever] ──── searches ────► [Vector Store (ChromaDB)]
     │                                    ▲
     │                           [Embeddings stored here]
     │                                    ▲
     │                           [Document Chunks]
     │                                    ▲
     │                           [Raw Documents (PDFs, etc.)]
     │
     ▼
[Context + Question] ──► [LLM (Google Gemini)] ──► Answer





## 📁 Project Structure



rag-document-assistant/
│
├── app.py                  # Entry point — runs the interactive Q&A loop
├── requirements.txt        # All Python dependencies (pinned versions)
├── test_imports.py         # Quick sanity check for your environment
│
├── data/                   # 📂 Put your source documents here (PDFs, etc.)
│
├── loaders/                # Loads raw documents from disk
│   ├── __init__.py
│   └── loader.py
│
├── chunking/               # Splits documents into smaller chunks
│   ├── __init__.py
│   └── chunking.py
│
├── embeddings/             # Converts text chunks into vector embeddings
│   └── __init__.py
│
├── vectorstore/            # Stores and indexes embeddings using ChromaDB
│   ├── __init__.py
│   └── vectorstore.py
│
├── retrieval/              # Retrieves the most relevant chunks for a query
│   ├── __init__.py
│   └── retriver.py
│
├── llm/                    # Wraps the LLM (Google Gemini) for generation
│   ├── __init__.py
│   └── llm.py
│
└── pipeline/               # Combines retriever + LLM into one callable pipeline
    ├── __init__.py
    └── rag_pipeline.py



> Each folder contains an `__init__.py` file, making every module a proper Python package. This is intentional — it keeps the codebase upgrade-friendly, allowing any module to be imported cleanly elsewhere, extended with new classes, or published as a standalone package in the future without restructuring the project.

Each folder is a self-contained module — you can modify any single stage (e.g., swap ChromaDB for FAISS, or Gemini for OpenAI) without touching the rest.




## ⚙️ How It Works — Step by Step

**Indexing phase** (happens once, when you first run the pipeline):

1. **Load** — `loaders/loader.py` reads documents from the `data/` folder (PDFs, text files, etc.)
2. **Chunk** — `chunking/chunking.py` splits them into smaller, overlapping text chunks for better retrieval
3. **Embed** — Sentence Transformers convert each chunk into a dense vector representation
4. **Store** — ChromaDB stores and indexes those vectors in `vectorstore/`

**Query phase** (every time you ask a question):

5. **Retrieve** — `retrieval/retriver.py` embeds your question and finds the most similar chunks in ChromaDB
6. **Generate** — `pipeline/rag_pipeline.py` passes the retrieved context + your question to Google Gemini via `llm/llm.py`
7. **Answer** — The response is printed to your terminal




## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- A Google Gemini API key ([get one free here](https://aistudio.google.com/app/apikey))
- Git




### 1. Clone the Repository



git clone https://github.com/Surya117117/rag-document-assistant.git
cd rag-document-assistant





### 2. Create and Activate a Virtual Environment

**Windows:**


python -m venv .venv
.venv\Scripts\activate


**macOS / Linux:**


python -m venv .venv
source .venv/bin/activate




### 3. Install Dependencies

pip install -r requirements.txt


> ⚠️ The `requirements.txt` includes pinned versions for reproducibility. Installation may take a few minutes — packages like `torch`, `sentence-transformers`, and `chromadb` are large.



### 4. Set Up Your API Key

Create a `.env` file in the project root:


# .env
GOOGLE_API_KEY=your_gemini_api_key_here


> The project uses `python-dotenv` to load this automatically — never commit your `.env` file.



### 5. Add Your Documents

Place any PDF or text files you want to query into the `data/` folder:


data/
├── my_report.pdf
├── research_paper.pdf
└── notes.txt




### 6. Run the App


python app.py


You'll see an interactive prompt:


Ask Question: What are the key findings in the research paper?

Answer:
Based on the document, the key findings are...

Ask Question: exit


Type `exit` to quit.



## 🔍 Verifying Your Setup

Before running `app.py`, you can verify all imports work correctly:


python test_imports.py


This checks that LangChain, ChromaDB, Sentence Transformers, and Google Gemini are all importable in your environment.



## 🛠️ Customisation Guide

The modular design makes it easy to adapt the pipeline to your needs:

| What you want to change | Where to look |
|-------------------------|---------------|
| Support different file types | `loaders/loader.py` |
| Change chunk size / overlap | `chunking/chunking.py` |
| Use a different embedding model | `embeddings/` |
| Swap ChromaDB for FAISS or Pinecone | `vectorstore/vectorstore.py` |
| Use OpenAI or Anthropic instead of Gemini | `llm/llm.py` |
| Change the prompt template | `pipeline/rag_pipeline.py` |
| Add a web UI (Streamlit, Gradio, etc.) | Replace `app.py` |



## 📦 Key Dependencies

| Package | Purpose |
|---------|---------|
| `langchain` | Core orchestration framework |
| `langchain-google-genai` | Google Gemini integration |
| `chromadb` | Local vector store |
| `sentence-transformers` | Local embedding model |
| `PyMuPDF` | PDF text extraction |
| `python-dotenv` | Environment variable management |
| `torch` | Required by Sentence Transformers |



## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.



## 📄 License

This project is licensed under the **Apache 2.0 License** — see the [LICENSE](LICENSE) file for details.



## 👤 Author

**Surya Pratap Singh** — [GitHub @Surya117117](https://github.com/Surya117117)
