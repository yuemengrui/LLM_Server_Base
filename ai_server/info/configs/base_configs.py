# *_*coding:utf-8 *_*
# @Author : YueMengRui

FASTAPI_HOST = '0.0.0.0'
FASTAPI_PORT = 5000

LLM_HISTORY_LEN = 10
EMBEDDING_ENCODE_BATCH_SIZE = 8
LLM_MODEL_LIST = [
    {
        "model_type": "ChatGLM",
        "model_name": "ChatGLM2_6B",
        "embedding_dim": 4096,
        "model_name_or_path": "",
        "device": "cuda"
    },
    {
        "model_type": "Baichuan",
        "model_name": "Baichuan_13B",
        "embedding_dim": 5120,
        "model_name_or_path": "",
        "device": "cuda"
    }
]

EMBEDDING_MODEL_LIST = [
    {
        "embedding_type": "text",
        "model_name": "m3e_base",
        "max_seq_length": 512,
        "embedding_dim": 768,
        "model_name_or_path": "",
        "device": "cuda"
    },
    {
        "embedding_type": "text",
        "model_name": "text2vec_large_chinese",
        "max_seq_length": 512,
        "embedding_dim": 1024,
        "model_name_or_path": "",
        "device": "cuda"
    }
]

# API LIMIT
API_LIMIT = {
    "model_list": "120/minute",
    "chat": "15/minute",
    "token_count": "60/minute",
    "text_embedding": "60/minute",
}
