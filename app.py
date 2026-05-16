from pipeline.rag_pipeline import retriever, llm

while True:

    query = input("Ask Question: ")

    if query == "exit":
        break

    docs = retriever.retrieve(query)

    context = "\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
    Context:
    {context}

    Question:
    {query}
    """

    response = llm.invoke(prompt)

    print("\nAnswer:\n")
    print(response.content)