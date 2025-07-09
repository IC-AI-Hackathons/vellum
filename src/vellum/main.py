from vellum.documents.store import Documents
from PIL import Image

from vellum.documents.images import extract_images, split_document_pages

if __name__ == "__main__":
    # Load images from devito.pdf
    image_uris = split_document_pages('assets/devito.pdf')
    print("images:", image_uris)
    # images = extract_images('assets/devito.pdf')
    # for
