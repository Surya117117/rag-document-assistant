class QueryGenerator:

    def __init__(self, llm):
        self.llm = llm

    def generate_queries(self, query):
        prompt = f"""
        Generate 4 different rephrased versions of the user query
        for better semantic retrieval.

        Original Query:
        {query}

        Return ONLY the queries separated by new lines.
        """

        response = self.llm.invoke(prompt)

        queries = response.content.split("\n")

        queries = [
            q.strip("- ").strip()
            for q in queries
            if q.strip()
        ]

        return queries