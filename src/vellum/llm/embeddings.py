from colpali_engine.models import BiQwen2_5, BiQwen2_5_Processor
from transformers.utils.import_utils import is_flash_attn_2_available
from PIL.Image import Image, open as open_image
import torch


__all__ = ['BiQwenEmbeddings', 'colqwen_embeddings']


class BiQwenEmbeddings:
    """"
    Provides embeddings with ColQwen2.5.
    """

    def __init__(self, model_name: str) -> None:
        self.model = BiQwen2_5.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map='cuda:0',
            attn_implementation=('flash_attention_2'
                                 if is_flash_attn_2_available() else None))

        self.processor = BiQwen2_5_Processor.from_pretrained(
            model_name,
            use_fast=True)

    def _embed_image_objects(self, images: list[Image]) -> list[list[float]]:
        batch_size = 4
        result: list[list[float]] = [None] * len(images)
        for i in range(0, len(images), batch_size):
            # Process images in batches
            batch = images[i:min(i + batch_size, len(images))]
            batch_images = self.processor \
                .process_images(batch) \
                .to(self.model.device)

            with torch.no_grad():
                image_embeddings: torch.Tensor = self.model(**batch_images)
                for j, embedding in enumerate(image_embeddings \
                                              .float().cpu().numpy()):
                    result[i + j] = embedding

        return result

    def embed_image(self, uris: list[str]) -> list[list[float]]:
        images = [open_image(uri) for uri in uris]
        return self._embed_image_objects(images)

    def embed_query(self, query: str) -> list[float]:
        batch_query = self.processor \
            .process_queries([query]) \
            .to(self.model.device)

        with torch.no_grad():
            query_embedding: torch.Tensor = self.model(**batch_query)[0]
            return query_embedding.float().cpu().numpy()


embeddings_model = 'nomic-ai/nomic-embed-multimodal-3b'
colqwen_embeddings = BiQwenEmbeddings(embeddings_model)
