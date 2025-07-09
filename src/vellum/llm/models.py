from colpali_engine.models import ColQwen2_5, ColQwen2_5Processor
from transformers.utils.import_utils import is_flash_attn_2_available
import torch


__all__ = ['embeddings_model', 'embeddings_processor']


embeddings_model_name = 'nomic-ai/colnomic-embed-multimodal-3b'

# Multimodal embeddings model
embeddings_model = ColQwen2_5.from_pretrained(
    embeddings_model_name,
    torch_dtype=torch.bfloat16,
    device_map='auto',
    attn_implementation=('flash_attention_2'
                         if is_flash_attn_2_available() else None))

# Multimodal processor
embeddings_processor = ColQwen2_5Processor.from_pretrained(
    embeddings_model_name)
