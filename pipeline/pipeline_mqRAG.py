from loaders.loader import load_documents
from chunking.chunking import split_documents
from embeddings.embedding_manager import EmbeddingManager
from vectorstore.vectorstore import VectorStore
from retrievers.multi_query_retriever import MultiQueryRetriever
from llm.llm import load_llm
from llm.multi_query_generator import QueryGenerator

documents = load_documents(
    r"D:\Tech Stack\Modular RAG Pipelines\data"
)

chunks = split_documents(documents)
embedding_manager = EmbeddingManager()

texts = [doc.page_content for doc in chunks]
embeddings = embedding_manager.generate_embeddings(texts)

vectorstore = VectorStore()
vectorstore.add_documents(
    chunks,
    embeddings
)

llm = load_llm()
query_generator = QueryGenerator(llm)

retriever = MultiQueryRetriever(
    vectorstore=vectorstore,
    embedding_manager=embedding_manager,
    query_generator=query_generator
)