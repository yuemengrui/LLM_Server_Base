# *_*coding:utf-8 *_*
# @Author : YueMengRui
import json
from info import llm_dict, logger
from info.modules.protocol import ChatResponse


def token_counter(model_name: str, prompt: str):
    return llm_dict[model_name]['model'].check_token_len(prompt)


def llm_generate(model_name, prompt, history, stream, history_len=10, **kwargs):
    if stream:
        def stream_generator():
            for resp in llm_dict[model_name]['model'].lets_chat(prompt, history, stream=True, **kwargs):
                resp.update({"model_name": model_name})
                resp["history"] = resp["history"][-history_len:]
                yield json.dumps(resp, ensure_ascii=False)
            logger.info(str(resp) + '\n')

        return stream_generator()

    else:
        resp = llm_dict[model_name]['model'].lets_chat(prompt, history, stream=False, **kwargs)

        resp.update({"model_name": model_name})
        resp["history"] = resp["history"][-history_len:]
        logger.info(str(resp) + '\n')
        return ChatResponse(**resp)
