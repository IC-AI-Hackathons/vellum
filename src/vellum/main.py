import torch
from vellum.llm.models import embeddings_model, embeddings_processor
from PIL import Image

from vellum.documents.images import extract_images

if __name__ == "__main__":
    # Load images from devito.pdf
    print("Extracting images...")
    images = extract_images('assets/devito.pdf')
    print(f"Loaded {len(images)} images")

    batch_size = 8
    for i in range(0, len(images), batch_size):
        batch = images[i:min(i + batch_size, len(images))]

        print("Processing images...")
        batch_images = embeddings_processor.process_images(batch) \
            .to(embeddings_model.device)

        print("Getting embeddings...")
        with torch.no_grad():
            image_embeddings = embeddings_model(**batch_images)

        print(image_embeddings)
