# *_*coding:utf-8 *_*
# @Author : YueMengRui
from info.configs import *
from pydantic import BaseModel, Field
from typing import Dict, List


class ErrorResponse(BaseModel):
    errcode: int
    errmsg: str


class ModelListResponse(BaseModel):
    errcode: int
    errmsg: str
    data: dict = {"model_list": []}


class BaseResponse(BaseModel):
    errcode: int
    errmsg: str
    data: Dict = {}


class ChatRequest(BaseModel):
    model_name: str = Field(default=None, description="模型名称")
    prompt: str
    history: List = Field(default=[], description="历史记录")
    generation_configs: Dict = {"history_len": LLM_HISTORY_LEN}
    stream: bool = Field(default=True, description="是否流式输出")


class ChatResponse(BaseModel):
    model_name: str
    answer: str
    usage: Dict


class EmbeddingRequest(BaseModel):
    model_name: str = Field(default=None, description="模型名称")
    sentences: List[str] = Field(description="句子列表")


class TokenCountRequest(BaseModel):
    model_name: str = Field(default=None, description="模型名称")
    prompt: str
