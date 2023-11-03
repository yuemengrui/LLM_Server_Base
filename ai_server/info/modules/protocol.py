# *_*coding:utf-8 *_*
# @Author : YueMengRui
from pydantic import BaseModel, Field
from typing import Dict, List


class ErrorResponse(BaseModel):
    object: str = "error"
    errcode: int
    errmsg: str


class ModelCard(BaseModel):
    model_name: str


class ModelListResponse(BaseModel):
    object: str = "llm_model_list"
    data: List[ModelCard] = []


class ChatRequest(BaseModel):
    model_name: str
    prompt: str
    history: List = Field(default=[], description="历史记录")
    generation_configs: Dict = {}
    stream: bool = Field(default=True, description="是否流式输出")


class TokenCountRequest(BaseModel):
    model_name: str
    prompt: str
