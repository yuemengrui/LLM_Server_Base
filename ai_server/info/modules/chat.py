# *_*coding:utf-8 *_*
# @Author : YueMengRui
import requests
from mylogger import logger
from fastapi import APIRouter, Request
from info import llm_dict, llm_name_list, limiter
from configs import API_LIMIT
from .protocol import ChatRequest, TokenCountRequest, ModelListResponse, ErrorResponse, ModelCard
from fastapi.responses import JSONResponse, StreamingResponse
from info.utils.response_code import RET, error_map

router = APIRouter()


@router.api_route(path='/ai/llm/list', methods=['GET'], response_model=ModelListResponse, summary="获取支持的llm列表")
@limiter.limit(API_LIMIT['model_list'])
def support_llm_list(request: Request):
    model_cards = []
    for i in llm_name_list:
        model_cards.append(ModelCard(model_name=i))
    return JSONResponse(ModelListResponse(data=model_cards).dict())


@router.api_route('/ai/llm/chat', methods=['POST'], summary="Chat")
@limiter.limit(API_LIMIT['chat'])
def llm_chat(request: Request,
             req: ChatRequest
             ):
    logger.info(req.dict())

    if req.model_name not in llm_name_list:
        return JSONResponse(ErrorResponse(errcode=RET.PARAMERR, errmsg=error_map[RET.PARAMERR]).dict(),
                            status_code=412)

    if req.stream:
        resp = requests.post(url=llm_dict[req.model_name]['chat'], json=req.dict(), stream=True)
        if 'event-stream' in resp.headers.get('content-type'):
            def stream_generate():
                for line in resp.iter_content(chunk_size=None):
                    yield line

            return StreamingResponse(stream_generate(), media_type="text/event-stream")
        else:
            return resp

    else:
        resp = requests.post(url=llm_dict[req.model_name]['chat'], json=req.dict())

        return resp


@router.api_route('/ai/llm/token_count', methods=['POST'], summary="token count")
@limiter.limit(API_LIMIT['token_count'])
def count_token(request: Request,
                req: TokenCountRequest
                ):
    logger.info(req.dict())

    if req.model_name not in llm_name_list:
        return JSONResponse(ErrorResponse(errcode=RET.PARAMERR, errmsg=error_map[RET.PARAMERR]).dict(),
                            status_code=412)

    resp = requests.post(url=llm_dict[req.model_name]['token_count'], json=req.dict())

    return resp
