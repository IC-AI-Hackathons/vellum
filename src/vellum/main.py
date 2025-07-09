from vellum.documents.store import Documents
from PIL import Image

from vellum.documents.images import extract_images, split_document_pages
from vellum.llm.chat import chat_model

if __name__ == "__main__":
    documents = Documents()
    documents.add_document('assets/devito.pdf')

    while True:
        print("\n\n> ", end='')
        query = input()
        if query == '/quit':
            break

        relevant_pages = documents.vector_store.similarity_search(query=query, k=1)
        for page in relevant_pages:
            print(f"Relevant page: {page.metadata['document']} (p. {page.metadata['page']})")
            # image = Image.open(page.page_content)
            # image.show()

