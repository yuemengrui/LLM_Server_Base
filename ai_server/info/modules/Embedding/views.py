# *_*coding:utf-8 *_*
# @Author : YueMengRui
import json
from . import embedding_blu
from copy import deepcopy
from info import limiter, llm_dict
from flask import request, jsonify, current_app
from info.utils.response_code import RET, error_map


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

    if model_name is None:
        model_name = list(llm_dict.keys())[0]

    res = []

    llm_conf = deepcopy(llm_dict[model_name])

    try:
        model = llm_conf.pop('model')
        embeddings = model.get_embeddings(sentences)
        embeddings = [x.tolist() for x in embeddings]
        llm_conf.update({"embeddings": embeddings})
        res.append(llm_conf)
    except Exception as e:
        current_app.logger.error(str({'EXCEPTION': e}) + '\n')

    return jsonify(errcode=RET.OK, errmsg=error_map[RET.OK], data=res)
