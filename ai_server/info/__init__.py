# *_*coding:utf-8 *_*
import os
import time
from info.configs import *
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from copy import deepcopy
from info.utils.logger import MyLogger
from sentence_transformers import SentenceTransformer

app = FastAPI(title="LLM_Server_Base")

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

logger = MyLogger()


@app.middleware("http")
async def log_requests(request, call_next):
    logger.info(f"start request {request.method} {request.url.path}")
    start = time.time()

    response = await call_next(request)

    cost = time.time() - start
    logger.info(f"end request {request.method} {request.url.path} {cost:.3f}s")
    return response


from info.libs.ai import build_model

llm_dict = {}
for llm_config in deepcopy(LLM_MODEL_LIST):
    if os.path.exists(llm_config['model_name_or_path']):
        llm = build_model(logger=logger, **llm_config)

        llm_dict[llm_config['model_name']] = {'model_name': llm_config['model_name'],
                                              'embedding_dim': llm_config['embedding_dim'], 'model': llm}

embedding_model_dict = {}
for embedding_config in deepcopy(EMBEDDING_MODEL_LIST):
    model_name_or_path = embedding_config.pop('model_name_or_path')
    device = embedding_config.pop('device')

    if model_name_or_path:
        embedding_model = SentenceTransformer(model_name_or_path=model_name_or_path, device=device)

        embedding_config.update({"model": embedding_model})
        embedding_model_dict[embedding_config['model_name']] = embedding_config

from info.modules import register_router

register_router(app)
