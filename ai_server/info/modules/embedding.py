# *_*coding:utf-8 *_*
# @Author : YueMengRui
from fastapi import APIRouter, Request
from copy import deepcopy
from sklearn.preprocessing import normalize
from info.configs.base_configs import API_LIMIT, EMBEDDING_ENCODE_BATCH_SIZE
from info import llm_dict, embedding_model_dict, logger, limiter
from fastapi.responses import JSONResponse
from .protocol import EmbeddingRequest, ModelListResponse, BaseResponse
from info.utils.response_code import RET, error_map

router = APIRouter()


@router.api_route(path='/ai/embedding/model/list', methods=['GET'], response_model=ModelListResponse,
                  summary="获取支持的embedding模型列表")
@limiter.limit(API_LIMIT['model_list'])
async def support_embedding_model_list(request: Request):
    res = []
    res.extend(list(embedding_model_dict.keys()))
    res.extend(list(llm_dict.keys()))
    return JSONResponse(ModelListResponse(errcode=RET.OK, errmsg=error_map[RET.OK], data={"model_list": res}).dict())


@router.api_route(path='/ai/embedding/text', methods=['POST'], response_model=BaseResponse, summary="文本embedding")
@limiter.limit(API_LIMIT['text_embedding'])
async def text_embedding(embedding_req: EmbeddingRequest, request: Request):
    logger.info(str(embedding_req.dict()))
    embedding_model_name_list = []
    embedding_model_name_list.extend(list(llm_dict.keys()))
    embedding_model_name_list.extend(list(embedding_model_dict.keys()))
    if embedding_req.model_name is None or embedding_req.model_name not in embedding_model_name_list:
        embedding_req.model_name = embedding_model_name_list[0]

    res = {}

    if embedding_req.model_name in llm_dict:
        llm_conf = llm_dict[embedding_req.model_name]
        temp = {}
        try:
            embeddings = llm_conf['model'].get_embeddings(embedding_req.sentences)
            embeddings = [x.tolist() for x in embeddings]
            temp.update({"embeddings": embeddings})
            temp.update({k: v for k, v in llm_conf.items() if k != 'model'})
            res = deepcopy(temp)
        except Exception as e:
            logger.error(str({'EXCEPTION': e}) + '\n')

    elif embedding_req.model_name in embedding_model_dict:
        embedding_model_config = embedding_model_dict[embedding_req.model_name]
        temp = {}

        try:
            embeddings = embedding_model_config['model'].encode(sentences=embedding_req.sentences,
                                                                batch_size=EMBEDDING_ENCODE_BATCH_SIZE)
            embeddings = normalize(embeddings, norm='l2')
            embeddings = [x.tolist() for x in embeddings]
            temp.update({"embeddings": embeddings})
            temp.update({k: v for k, v in embedding_model_config.items() if k != 'model'})
            res = deepcopy(temp)
        except Exception as e:
            logger.error(str({'EXCEPTION': e}) + '\n')

    return JSONResponse(BaseResponse(errcode=RET.OK, errmsg=error_map[RET.OK], data=res).dict())
