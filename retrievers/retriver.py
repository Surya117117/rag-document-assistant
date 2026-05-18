from typing import List, Dict, Any, Tuple
from embeddings.embedding_manager import EmbeddingManager
from vectorstore.vectorstore import VectorStore
from langchain_core.documents import Document

class RAGRetriever:
    """Handles query-based retrieval from the vector store"""

    def __init__(self, vectorstore: VectorStore, embedding_manager: EmbeddingManager):
        """Initialize the retriever
        Args:
            vectorstore: vector store containing document embeddings
            embedding_manager: Manager for generating query embeddings"""
        
        self.vectorstore = vectorstore
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query
        
        Args:
            query: The search query
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold
        
        Returns:
            List of dictionaries containing documents and metadata"""
        
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top k: {top_k}, Score threshold: {score_threshold}")

        #Generate query embeddings
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        #search in vector store
        try:
            results = self.vectorstore.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results = top_k
            )
        
            retrieved_docs = []

            if results ['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]

                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    #Converting distance to similarity score
                    similarity_score = 1 - distance

                    if similarity_score >= score_threshold:
                        retrieved_docs.append(
                        Document(
                            page_content=document,
                            metadata={
                                'id': doc_id,
                                'similarity_score': similarity_score,
                                'distance': distance,
                                'rank': i + 1
                            }
                        )
                    )
                
                print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")

            else:
                print("No documents found")
            
            return retrieved_docs
        
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []
        