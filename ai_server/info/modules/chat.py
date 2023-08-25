# *_*coding:utf-8 *_*
# @Author : YueMengRui
from fastapi import APIRouter, Request
from info import llm_dict, logger, limiter
from info.configs.base_configs import API_LIMIT
from .protocol import ChatRequest, TokenCountRequest
from fastapi.responses import JSONResponse, StreamingResponse
from info.utils.response_code import RET, error_map
from info.utils.llm_common import llm_generate, token_counter

router = APIRouter()


@router.api_route(path='/ai/llm/list', methods=['GET'], summary="获取支持的llm列表")
@limiter.limit(API_LIMIT['model_list'])
def support_llm_list(request: Request):
    return JSONResponse({"errcode": RET.OK, "errmsg": error_map[RET.OK], "data": {"llm_list": list(llm_dict.keys())}})


@router.api_route('/ai/llm/chat', methods=['POST'], summary="Chat")
@limiter.limit(API_LIMIT['chat'])
def llm_chat(request: ChatRequest):
    logger.info(str(request.dict()))

    model_name_list = list(llm_dict.keys())
    if request.model_name is None or request.model_name not in model_name_list:
        request.model_name = model_name_list[0]

    token_counter_resp = token_counter(request.model_name, request.prompt)
    if not token_counter_resp[0]:
        return JSONResponse({"errcode": RET.TOKEN_OVERFLOW,
                             "errmsg": error_map[RET.TOKEN_OVERFLOW] + u"当前prompt token:{} 支持的最大token:{}".format(
                                 token_counter_resp[1], token_counter_resp[2]), "data": {}})

    if request.stream:
        return StreamingResponse(llm_generate(model_name=request.model_name,
                                              prompt=request.prompt,
                                              history=request.history,
                                              stream=True,
                                              **request.generation_configs), media_type="text/event-stream")

    else:
        resp = llm_generate(model_name=request.model_name,
                            prompt=request.prompt,
                            history=request.history,
                            stream=False,
                            **request.generation_configs)

        return JSONResponse({"errcode": RET.OK, "errmsg": error_map[RET.OK], "data": resp.dict()})


@router.api_route('/ai/llm/token_count', methods=['POST'], summary="token count")
@limiter.limit(API_LIMIT['token_count'])
def count_token(request: TokenCountRequest):
    logger.info(str(request.dict()))

    model_name_list = list(llm_dict.keys())
    if request.model_name is None or request.model_name not in model_name_list:
        request.model_name = model_name_list[0]

    token_counter_resp = token_counter(request.model_name, request.prompt)

    return JSONResponse({"errcode": RET.OK, "errmsg": error_map[RET.OK],
                         "data": {"model_name": request.model_name, "prompt": request.prompt,
                                  "prompt_tokens": token_counter_resp[1]}})
