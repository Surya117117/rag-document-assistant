from pipeline.pipeline_mqRAG import retriever, llm

while True:
    query = input("\nAsk Question: ")
    if query.lower() == "exit":
        break

    docs, queries = retriever.retrieve(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )
    prompt = f"""
You are a helpful AI assistant.
Answer ONLY from the given context.

Context:
{context}

Question:
{query}

Answer:
"""
    response = llm.invoke(prompt)
    print("\nAnswer:\n")
    print(response.content)