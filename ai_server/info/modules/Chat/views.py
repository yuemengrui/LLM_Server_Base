# *_*coding:utf-8 *_*
# @Author : YueMengRui
import json
from . import chat_blu
from info import limiter, llm_dict
from flask import request, jsonify, current_app, Response
from info.utils.response_code import RET, error_map
from info.utils.llm_common import llm_stream_generate, llm_generate


@chat_blu.route('/ai/llm/list', methods=['GET'])
@limiter.limit("60 per minute", override_defaults=False)
def support_llm_list():
    return jsonify(errcode=RET.OK, errmsg=error_map[RET.OK], data={'llm_list': list(llm_dict.keys())})


@chat_blu.route('/ai/llm/chat', methods=['POST'])
@limiter.limit("15 per minute", override_defaults=False)
def llm_chat():
    data = request.get_data()
    json_data = json.loads(data.decode("utf-8"))
    queries = json_data.get('queries', [])
    model_name = json_data.get('model_name', None)

    current_app.logger.info(str({'model_name': model_name, 'queries': queries}) + '\n')

    if not queries:
        return jsonify(errcode=RET.PARAMERR, errmsg=error_map[RET.PARAMERR])

    model_name_list = list(llm_dict.keys())
    if model_name is None or model_name not in model_name_list:
        model_name = model_name_list[0]

    base_generation_configs = {"history_len": current_app.config['LLM_HISTORY_LEN']}
    prompt_list = []
    history_list = []
    for query_dict in queries:
        prompt = query_dict.get('prompt', '')
        history = query_dict.get('history', [])
        generation_configs = query_dict.get('generation_configs', {})

        if isinstance(generation_configs, dict):
            base_generation_configs.update(generation_configs)

        prompt_list.append(prompt)
        history_list.append(history)

    responses = llm_generate(model_name, prompt_list, history_list, **base_generation_configs)

    return jsonify(errcode=RET.OK, errmsg=error_map[RET.OK], data={'answers': responses})


@chat_blu.route('/ai/llm/stream/chat', methods=['POST'])
@limiter.limit("15 per minute", override_defaults=False)
def llm_chat_stream():
    data = request.get_data()
    json_data = json.loads(data.decode("utf-8"))
    queries = json_data.get('queries', [])
    model_name = json_data.get('model_name', None)

    current_app.logger.info(str({'model_name': model_name, 'queries': queries}) + '\n')

    if not queries:
        return jsonify(errcode=RET.PARAMERR, errmsg=error_map[RET.PARAMERR])

    model_name_list = list(llm_dict.keys())
    if model_name is None or model_name not in model_name_list:
        model_name = model_name_list[0]

    base_generation_configs = {"history_len": current_app.config['LLM_HISTORY_LEN']}
    prompt_list = []
    history_list = []
    for query_dict in queries:
        prompt = query_dict.get('prompt', '')
        history = query_dict.get('history', [])
        generation_configs = query_dict.get('generation_configs', {})

        if isinstance(generation_configs, dict):
            base_generation_configs.update(generation_configs)

        prompt_list.append(prompt)
        history_list.append(history)

    return Response(llm_stream_generate(model_name, prompt_list, history_list, **base_generation_configs),
                    mimetype='text/event-stream')
