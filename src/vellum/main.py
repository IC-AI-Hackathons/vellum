from vellum.documents.store import Documents
from PIL import Image

from vellum.documents.images import extract_images, split_document_pages

if __name__ == "__main__":
    documents = Documents()
    documents.add_document('assets/devito.pdf')

    while True:
        print("\n> ", end='')
        query = input()
        if query == '/quit':
            break

        res = documents.vector_store.similarity_search(query=query, k=3)
        from pprint import pprint
        pprint(res)
