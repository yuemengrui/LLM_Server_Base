# *_*coding:utf-8 *_*
# @Author : YueMengRui
import os

FASTAPI_HOST = os.getenv('FASTAPI_HOST', '0.0.0.0')
FASTAPI_PORT = os.getenv('FASTAPI_PORT', 5000)

LLM_HISTORY_LEN = 10

LLM_MODEL_LIST = [
    {
        "model_type": "ChatGLM",
        "model_name": "ChatGLM2-6B",
        "embedding_dim": 4096,
        "model_name_or_path": "",
        "device": "cuda"
    },
    {
        "model_type": "Baichuan",
        "model_name": "Baichuan-13B",
        "embedding_dim": 5120,
        "model_name_or_path": "",
        "device": "cuda"
    }
]

EMBEDDING_MODEL_LIST = [
    {
        "embedding_type": "text",
        "model_name": "m3e-base",
        "max_seq_length": 512,
        "embedding_dim": 768,
        "model_name_or_path": "",
        "device": "cuda"
    },
    {
        "embedding_type": "text",
        "model_name": "text2vec-large-chinese",
        "max_seq_length": 512,
        "embedding_dim": 1024,
        "model_name_or_path": "",
        "device": "cuda"
    }
]
