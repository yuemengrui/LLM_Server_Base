# *_*coding:utf-8 *_*
# @Author : YueMengRui
from fastapi import APIRouter, Request
from info import llm_dict, logger, limiter
from info.configs.base_configs import API_LIMIT
from .protocol import ChatRequest, TokenCountRequest, ModelListResponse
from fastapi.responses import JSONResponse, StreamingResponse
from info.utils.response_code import RET, error_map
from info.utils.llm_common import llm_generate, token_counter

router = APIRouter()


@router.api_route(path='/ai/llm/list', methods=['GET'], response_model=ModelListResponse, summary="获取支持的llm列表")
@limiter.limit(API_LIMIT['model_list'])
def support_llm_list(request: Request):
    return JSONResponse(
        ModelListResponse(errcode=RET.OK, errmsg=error_map[RET.OK], data={"model_list": list(llm_dict.keys())}).dict())


@router.api_route('/ai/llm/chat', methods=['POST'], summary="Chat")
@limiter.limit(API_LIMIT['chat'])
def llm_chat(chat_req: ChatRequest, request: Request):
    logger.info(str(chat_req.dict()))

    model_name_list = list(llm_dict.keys())
    if chat_req.model_name is None or chat_req.model_name not in model_name_list:
        chat_req.model_name = model_name_list[0]

    token_counter_resp = token_counter(chat_req.model_name, chat_req.prompt)
    if not token_counter_resp[0]:
        return JSONResponse({"errcode": RET.TOKEN_OVERFLOW,
                             "errmsg": error_map[RET.TOKEN_OVERFLOW] + u"当前prompt token:{} 支持的最大token:{}".format(
                                 token_counter_resp[1], token_counter_resp[2]), "data": {}})

    if chat_req.stream:
        return StreamingResponse(llm_generate(model_name=chat_req.model_name,
                                              prompt=chat_req.prompt,
                                              history=chat_req.history,
                                              stream=True,
                                              **chat_req.generation_configs), media_type="text/event-stream")

    else:
        resp = llm_generate(model_name=chat_req.model_name,
                            prompt=chat_req.prompt,
                            history=chat_req.history,
                            stream=False,
                            **chat_req.generation_configs)

        return JSONResponse({"errcode": RET.OK, "errmsg": error_map[RET.OK], "data": resp.dict()})


@router.api_route('/ai/llm/token_count', methods=['POST'], summary="token count")
@limiter.limit(API_LIMIT['token_count'])
def count_token(token_count_req: TokenCountRequest, request: Request):
    logger.info(str(token_count_req.dict()))

    model_name_list = list(llm_dict.keys())
    if token_count_req.model_name is None or token_count_req.model_name not in model_name_list:
        token_count_req.model_name = model_name_list[0]

    token_counter_resp = token_counter(token_count_req.model_name, token_count_req.prompt)

    return JSONResponse({"errcode": RET.OK, "errmsg": error_map[RET.OK],
                         "data": {"model_name": token_count_req.model_name, "prompt": token_count_req.prompt,
                                  "prompt_tokens": token_counter_resp[1]}})
