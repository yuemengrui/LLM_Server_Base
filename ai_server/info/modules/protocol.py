# *_*coding:utf-8 *_*
# @Author : YueMengRui
from info.configs import *
from pydantic import BaseModel, Field
from typing import Dict, List, Literal, Optional, Union


class ChatRequest(BaseModel):
    model_name: str = None
    prompt: str
    history: List[List[str, str]] = []
    generation_configs: Dict = {"history_len": LLM_HISTORY_LEN}
    stream: bool = True


class ChatResponse(BaseModel):
    model_name: str
    answer: str
    history: List[List[str, str]]
    usage: Dict


class EmbeddingRequest(BaseModel):
    model_name: str = None
    sentences: List[str]
