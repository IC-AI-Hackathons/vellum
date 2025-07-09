from langchain_chroma import Chroma
from langchain.storage import InMemoryByteStore

from vellum.documents.images import split_document_pages
from vellum.llm.embeddings import colqwen_embeddings


class Documents:
    def __init__(self, collection_name: str = "vellum_documents"):
        self.byte_store = InMemoryByteStore()
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=colqwen_embeddings)

    def add_document(self, file_name: str) -> None:
        image_uris = split_document_pages(file_name)
        image_meta = [
            {
                'uri': uri,
                'document': file_name,
                'page': i + 1
            }
            for i, uri in enumerate(image_uris)
        ]

        self.vector_store.add_images(
            uris=image_uris,
            metadata=image_meta)
