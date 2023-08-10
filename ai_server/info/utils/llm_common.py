# *_*coding:utf-8 *_*
# @Author : YueMengRui
import json
from info import llm_dict
from info.modules.protocol import ChatResponse


def llm_generate(model_name, prompt, history, history_len=10, stream=True, **kwargs):
    if stream:
        for resp in llm_dict[model_name]['model'].lets_chat(prompt, history, stream=True, **kwargs):
            resp.update({"model_name": model_name})
            resp["history"] = resp["history"][-history_len:]
            yield json.dumps(resp, ensure_ascii=False)
    else:
        resp = llm_dict[model_name]['model'].lets_chat(prompt, history, stream=False, **kwargs)

        resp.update({"model_name": model_name})
        resp["history"] = resp["history"][-history_len:]

        return ChatResponse(model_name=model_name, answer=resp)
