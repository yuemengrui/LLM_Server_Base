# *_*coding:utf-8 *_*
# @Author : YueMengRui
import sys
import requests
from info.configs import LLM_SERVER_PORT, LLM_SERVER_CHAT, LLM_SERVER_TOKEN_COUNT


def server_is_online(logger):
    llm_dict = {}
    req_data = {
        "prompt": "你好",
        "stream": False
    }
    for llm_name in LLM_SERVER_PORT:
        try:
            resp = requests.post(url=LLM_SERVER_CHAT.replace('{port}', str(LLM_SERVER_PORT[llm_name])), json=req_data)
            if resp.status_code == 200:
                llm_dict.update(
                    {llm_name: {"chat": LLM_SERVER_CHAT.replace('{port}', str(LLM_SERVER_PORT[llm_name])),
                                "token_count": LLM_SERVER_TOKEN_COUNT.replace('{port}', LLM_SERVER_PORT[llm_name])}})
            else:
                logger.error(f"{llm_name} server offline!!!  {resp.text}")
        except Exception as e:
            logger.error(f"{llm_name} server offline!!!  {e}")

    if llm_dict == {}:
        logger.error("no llm server online!!! Program Exit!!!")
        sys.exit()

    return llm_dict
