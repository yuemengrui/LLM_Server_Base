# *_*coding:utf-8 *_*
# @Author : YueMengRui
import json
from . import embedding_blu
from copy import deepcopy
from info import limiter, llm_dict, embedding_model_dict
from flask import request, jsonify, current_app
from info.utils.response_code import RET, error_map


@embedding_blu.route('/ai/embedding_model/list', methods=['GET'])
@limiter.limit("60 per minute", override_defaults=False)
def support_embedding_model_list():
    res = []
    res.extend(list(embedding_model_dict.keys()))
    res.extend(list(llm_dict.keys()))

    return jsonify(errcode=RET.OK, errmsg=error_map[RET.OK], data={'embedding_model_list': res})


@embedding_blu.route('/ai/llm/text/embedding', methods=['POST'])
@limiter.limit("60 per minute", override_defaults=False)
def text_embedding():
    data = request.get_data()
    json_data = json.loads(data.decode("utf-8"))
    sentences = json_data.get('sentences', [])
    model_name = json_data.get('model_name', None)

    current_app.logger.info(str({'model_name': model_name, 'sentences': sentences}) + '\n')

    if not sentences:
        return jsonify(errcode=RET.PARAMERR, errmsg=error_map[RET.PARAMERR])

    embedding_model_name_list = []
    embedding_model_name_list.extend(list(llm_dict.keys()))
    embedding_model_name_list.extend(list(embedding_model_dict.keys()))
    if model_name is None or model_name not in embedding_model_name_list:
        model_name = embedding_model_name_list[0]

    res = []

    if model_name in llm_dict:
        llm_conf = deepcopy(llm_dict[model_name])

        try:
            model = llm_conf.pop('model')
            embeddings = model.get_embeddings(sentences)
            embeddings = [x.tolist() for x in embeddings]
            llm_conf.update({"embeddings": embeddings})
            res.append(llm_conf)
        except Exception as e:
            current_app.logger.error(str({'EXCEPTION': e}) + '\n')

    elif model_name in embedding_model_dict:
        embedding_model_config = deepcopy(embedding_model_dict[model_name])

        try:
            model = embedding_model_config.pop('model')
            embeddings = model.encode(sentences)
            embeddings = [x.tolist() for x in embeddings]
            embedding_model_config.update({"embeddings": embeddings})
            res.append(embedding_model_config)
        except Exception as e:
            current_app.logger.error(str({'EXCEPTION': e}) + '\n')

    return jsonify(errcode=RET.OK, errmsg=error_map[RET.OK], data=res)
