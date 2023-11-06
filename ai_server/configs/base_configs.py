# *_*coding:utf-8 *_*
# @Author : YueMengRui
import os

FASTAPI_HOST = '0.0.0.0'
FASTAPI_PORT = 5000

LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)
# LLM Server
LLM_SERVER_PREFIX = "http://127.0.0.1"
LLM_SERVER_PORT = {
    "Baichuan2_13B_8k": 10000,
    "ChatGLM2_6B_32k": 10001,
    "InternLM_20B_16k": 10002
}

LLM_SERVER_CHAT = LLM_SERVER_PREFIX + ":{port}/ai/llm/chat"
LLM_SERVER_TOKEN_COUNT = LLM_SERVER_PREFIX + ":{port}/ai/llm/token_count"

# API LIMIT
API_LIMIT = {
    "model_list": "120/minute",
    "chat": "15/minute",
    "token_count": "60/minute"
}
