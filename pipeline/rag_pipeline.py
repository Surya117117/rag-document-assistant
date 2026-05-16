from loaders.loader import load_documents
from chunking.chunking import split_documents
from embeddings.embedding_manager import EmbeddingManager
from vectorstore.vectorstore import VectorStore
from retrieval.retriver import RAGRetriever
from llm.llm import load_llm


# Load documents
documents = load_documents("data")

# Split documents
chunks = split_documents(documents)

# Embedding model
embedding_manager = EmbeddingManager()

texts = [doc.page_content for doc in chunks]


embeddings = embedding_manager.generate_embeddings(texts)

# Vector database
vectorstore = VectorStore()

vectorstore.add_documents(chunks, embeddings)

# Retriever
retriever = RAGRetriever(
    vectorstore,
    embedding_manager
)

# LLM
llm = load_llm()