from typing import List
from langchain_core.documents import Document


class MultiQueryRetriever:

    def __init__(
        self,
        vectorstore,
        embedding_manager,
        query_generator
    ):

        self.vectorstore = vectorstore
        self.embedding_manager = embedding_manager
        self.query_generator = query_generator

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        score_threshold: float = 0.0
    ):
        print(f"\nOriginal Query: {query}")
        queries = self.query_generator.generate_queries(query)
        print("\nGenerated Queries:")

        for i, q in enumerate(queries, 1):
            print(f"{i}. {q}")

        all_docs = []

        for q in queries:
            print(f"\nRetrieving for: {q}")
            query_embedding = self.embedding_manager.generate_embeddings([q])[0]
            try:

                results = self.vectorstore.collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=top_k
                )
                if results["documents"] and results["documents"][0]:
                    documents = results["documents"][0]
                    metadatas = results["metadatas"][0]
                    distances = results["distances"][0]
                    ids = results["ids"][0]

                    for i, (
                        doc_id,
                        document,
                        metadata,
                        distance
                    ) in enumerate(
                        zip(ids, documents, metadatas, distances)
                    ):
                        similarity_score = 1 - distance
                        if similarity_score >= score_threshold:
                            all_docs.append(
                                Document(
                                    page_content=document,
                                    metadata={
                                        "id": doc_id,
                                        "similarity_score": similarity_score,
                                        "distance": distance,
                                        "rank": i + 1,
                                        "query": q
                                    }
                                )
                            )

            except Exception as e:
                print(f"Retrieval Error: {e}")

        unique_docs = list(
            {doc.page_content: doc for doc in all_docs}.values()
        )
        print(f"\nTotal unique documents retrieved: {len(unique_docs)}")

        return unique_docs, queries