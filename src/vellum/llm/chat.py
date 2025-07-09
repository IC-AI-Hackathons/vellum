import os
from dotenv import load_dotenv, find_dotenv
from langchain_ollama.chat_models import ChatOllama

__all__ = ['chat_model', 'chat_model']

load_dotenv(find_dotenv())
OLLAMA_BASE_URL = os.environ['OLLAMA_BASE_URL']
OLLAMA_API_KEY = os.environ['OLLAMA_API_KEY']

chat_model_id = 'qwen3:32b'
chat_model = ChatOllama(
    base_url=OLLAMA_BASE_URL,
    client_kwargs={
        'headers': {
            'Authorization': f"Bearer {OLLAMA_API_KEY}",
            'Content-Type': "application/json",
        }
    },
    model=chat_model_id,
    streaming=True)
