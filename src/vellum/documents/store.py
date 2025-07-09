from langchain_chroma import Chroma
from langchain.storage import InMemoryByteStore
# from langchain.retrievers import MultiVectorRetriever

from vellum.llm.embeddings import colqwen_embeddings


class Documents:
    def __init__(self, collection_name: str = "vellum_documents"):
        self.byte_store = InMemoryByteStore()
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=colqwen_embeddings)
