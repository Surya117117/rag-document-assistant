from langchain_community.document_loaders import (
    DirectoryLoader,
    PyMuPDFLoader
)


def load_documents(path):

    loader = DirectoryLoader(
        path,
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader,
        show_progress=False
    )

    documents = loader.load()

    print(f"Loaded {len(documents)} documents")

    return documents