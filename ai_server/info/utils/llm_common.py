# *_*coding:utf-8 *_*
# @Author : YueMengRui
import json
from info import llm_dict


def llm_stream_generate(model_name, prompt_list, history_list, history_len=10, **kwargs):
    for resp_list, history_list in llm_dict[model_name]['model'].lets_stream_chat(prompt_list, history_list, **kwargs):
        responses = []
        for i in range(len(resp_list)):
            history_list[i][-1][0] = prompt_list[i]
            history_list[i][-1][1] = resp_list[i]

            responses.append(
                {'model_name': model_name, 'answer': resp_list[i], 'history': history_list[i][-history_len:]})

        yield json.dumps({"answers": responses}, ensure_ascii=False)


def llm_generate(model_name, prompt_list, history_list, history_len=10, **kwargs):
    resp_list = llm_dict[model_name]['model'].letschat(prompt_list, history_list, **kwargs)
    responses = []

    for i in range(len(prompt_list)):
        history_list[i].append([prompt_list[i], resp_list[i]])

        responses.append({'model_name': model_name, 'answer': resp_list[i], 'history': history_list[i][-history_len:]})

    return responses
