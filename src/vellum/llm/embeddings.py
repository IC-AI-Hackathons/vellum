from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from transformers.utils.import_utils import is_flash_attn_2_available
from PIL import Image
import torch


__all__ = ['ColQwenEmbeddings', 'colqwn_embeddings']


class ColQwenEmbeddings:
    """"
    Provides embeddings with ColQwen2.5.
    """

    def __init__(self, model_name: str) -> None:
        self.model = ColQwen2_5.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map='cuda:0',
            attn_implementation=('flash_attention_2'
                                 if is_flash_attn_2_available() else None))

        self.processor = ColQwen2_5_Processor.from_pretrained(
            model_name,
            use_fast=True)

    def embed_image(self, image: Image) -> list[float]:
        batch = self.processor \
            .process_images([image]) \
            .to(self.model.device)

        with torch.no_grad():
            image_embeddings: torch.Tensor = self.model(**batch)
            return image_embeddings[0].flatten().tolist()


colqwen_embeddings_model = 'nomic-ai/colnomic-embed-multimodal-3b'
colqwen_embeddings = ColQwenEmbeddings(colqwen_embeddings_model)
