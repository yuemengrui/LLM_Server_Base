# *_*coding:utf-8 *_*
# @Author : YueMengRui
from fastapi import APIRouter
from info import llm_dict, logger
from .protocol import ChatRequest
from fastapi.responses import JSONResponse, StreamingResponse
from info.utils.response_code import RET, error_map
from info.utils.llm_common import llm_generate, llm_stream_generate

router = APIRouter()


@router.api_route(path='/ai/llm/list', methods=['GET'], summary="获取支持的llm列表")
def support_llm_list():
    return JSONResponse({"errcode": RET.OK, "errmsg": error_map[RET.OK], "data": {"llm_list": list(llm_dict.keys())}})


@router.api_route('/ai/llm/chat', methods=['POST'], summary="Chat")
def llm_chat(chat_request: ChatRequest):
    logger.info(str(chat_request.dict()) + '\n')

    model_name_list = list(llm_dict.keys())
    if chat_request.model_name is None or chat_request.model_name not in model_name_list:
        chat_request.model_name = model_name_list[0]

    if chat_request.stream:
        return StreamingResponse(llm_stream_generate(model_name=chat_request.model_name,
                                                     prompt=chat_request.prompt,
                                                     history=chat_request.history,
                                                     **chat_request.generation_configs), media_type="text/event-stream")

    else:
        resp = llm_generate(model_name=chat_request.model_name,
                            prompt=chat_request.prompt,
                            history=chat_request.history,
                            **chat_request.generation_configs)

        return JSONResponse(resp.json(ensure_ascii=False))
