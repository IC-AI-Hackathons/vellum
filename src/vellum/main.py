from vellum.documents.store import Documents
from PIL import Image

from vellum.documents.images import extract_images, split_document_pages

if __name__ == "__main__":
    documents = Documents()
    documents.add_document('assets/devito.pdf')

    while True:
        query = input("\n> ", end='')
        if query == '/quit':
            break

        res = documents.retriever.invoke(query)
        from pprint import pprint
        pprint(res)
