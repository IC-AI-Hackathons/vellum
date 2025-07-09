import torch
from vellum.llm.models import embeddings_model, embeddings_processor
from PIL import Image


if __name__ == "__main__":
    images = [
        Image.new('RGB', (100, 100), color='red'),
        Image.new('RGB', (100, 100), color='green'),
    ]

    print("processing images...")
    batch_images = embeddings_processor.process_images(images) \
        .to(embeddings_model.device)

    print("getting embeddings...")
    with torch.no_grad():
        image_embeddings = embeddings_model(**batch_images)

    print(image_embeddings)
