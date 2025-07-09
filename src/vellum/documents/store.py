from langchain_chroma import Chroma
from langchain.storage import InMemoryByteStore
from langchain.retrievers import MultiVectorRetriever

from vellum.documents.images import split_document_pages
from vellum.llm.embeddings import colqwen_embeddings


class Documents:
    def __init__(self, collection_name: str = "vellum_documents"):
        self.byte_store = InMemoryByteStore()
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=colqwen_embeddings)

        self.retriever = MultiVectorRetriever(
            vectorstore=self.vector_store,
            byte_store=self.byte_store,
            id_key='id')

    def add_document(self, file_name: str) -> None:
        image_uris = split_document_pages(file_name)
        image_meta = [
            {
                'id': f"{file_name}_{i}",
                'uri': uri,
                'document': file_name,
                'page': i
            }
            for i, uri in enumerate(image_uris)
        ]

        self.vector_store.add_images(
            uris=image_uris,
            metadata=image_meta,
            ids=[meta['id'] for meta in image_meta])

        self.byte_store.mset([
            (meta['id'], meta) for meta in image_meta])
