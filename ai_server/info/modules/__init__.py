# *_*coding:utf-8 *_*
from fastapi import FastAPI
from . import chat, embedding


def register_router(app: FastAPI):
    app.include_router(router=chat.router, prefix="", tags=["Chat"])
    app.include_router(router=embedding.router, prefix="", tags=['Embedding'])
