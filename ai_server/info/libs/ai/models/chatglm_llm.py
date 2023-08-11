# *_*coding:utf-8 *_*
# @Author : YueMengRui
import torch
from copy import deepcopy
import torch.nn.functional as F
from .base_model import BaseModel
from transformers import AutoTokenizer, AutoModel
from typing import Optional, Tuple, Union, List, Callable, Dict, Any


def torch_gc(device):
    if torch.cuda.is_available():
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def auto_configure_device_map(num_gpus: int) -> Dict[str, int]:
    # transformer.word_embeddings 占用1层
    # transformer.final_layernorm 和 lm_head 占用1层
    # transformer.layers 占用 28 层
    # 总共30层分配到num_gpus张卡上
    num_trans_layers = 28
    per_gpu_layers = 30 / num_gpus

    # bugfix: 在linux中调用torch.embedding传入的weight,input不在同一device上,导致RuntimeError
    # windows下 model.device 会被设置成 transformer.word_embeddings.device
    # linux下 model.device 会被设置成 lm_head.device
    # 在调用chat或者stream_chat时,input_ids会被放到model.device上
    # 如果transformer.word_embeddings.device和model.device不同,则会导致RuntimeError
    # 因此这里将transformer.word_embeddings,transformer.final_layernorm,lm_head都放到第一张卡上

    # device_map = {'transformer.word_embeddings': 0,
    #               'transformer.final_layernorm': 0, 'lm_head': 0}

    device_map = {
        'transformer.embedding.word_embeddings': 0,
        'transformer.encoder.final_layernorm': 0,
        'transformer.output_layer': 0,
        'transformer.rotary_pos_emb': 0,
        'lm_head': 0
    }

    used = 2
    gpu_target = 0
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0
        assert gpu_target < num_gpus
        # device_map[f'transformer.layers.{i}'] = gpu_target
        device_map[f'transformer.encoder.layers.{i}'] = gpu_target
        used += 1

    return device_map


class ChatGLM(BaseModel):

    def __init__(self, model_name_or_path, device='cuda', model_name=None, logger=None, **kwargs):
        self.model = None
        self.tokenizer = None
        self.device = device
        self.logger = logger
        self.max_length = 8192

        if model_name and '32k' in model_name:
            self.max_length = 32768

        self.max_prompt_length = self.max_length - 512
        self._load_model(model_name_or_path, device)

    def _load_model(self,
                    model_name_or_path,
                    device='cuda',
                    device_map: Optional[Dict[str, int]] = None,
                    **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )

        if torch.cuda.is_available() and device.lower().startswith("cuda"):
            # 根据当前设备GPU数量决定是否进行多卡部署
            num_gpus = torch.cuda.device_count()
            if num_gpus < 2 and device_map is None:
                self.model = (
                    AutoModel.from_pretrained(
                        model_name_or_path,
                        trust_remote_code=True,
                        **kwargs)
                    .half()
                    .cuda()
                )
            else:
                from accelerate import dispatch_model

                model = (
                    AutoModel.from_pretrained(
                        model_name_or_path,
                        trust_remote_code=True,
                        **kwargs)
                    .half())
                # 可传入device_map自定义每张卡的部署情况
                if device_map is None:
                    device_map = auto_configure_device_map(num_gpus)

                self.model = dispatch_model(model, device_map=device_map)
        else:
            if device == 'mps':
                self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).to('mps')

            else:
                self.model = (
                    AutoModel.from_pretrained(
                        model_name_or_path,
                        trust_remote_code=True,
                        **kwargs)
                    .float()
                    .to(device)
                )

        self.model = self.model.eval()

    def get_embeddings(self, sentences):
        embeddings = []
        for text in sentences:
            input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
            model_output = self.model(input_ids, output_hidden_states=True)
            data = (model_output.hidden_states[-1].transpose(0, 1))[0]
            data = F.normalize(torch.mean(data, dim=0), p=2, dim=0)
            embeddings.append(data)

        return embeddings

    def token_counter(self, prompt):
        return len(self.tokenizer(prompt, return_tensors="pt").input_ids[0])

    def select_history(self, prompt, history, max_prompt_length):
        base_prompt_token_num = self.token_counter("[Round 1]\n\n问：{}\n\n答：".format(prompt))
        true_history = []
        if history and base_prompt_token_num < max_prompt_length:
            for (old_query, old_response) in history[::-1]:
                history_token_num = self.token_counter(
                    "[Round 1]\n\n问：{}\n\n答：{}\n\n".format(old_query, old_response))
                if base_prompt_token_num + history_token_num > max_prompt_length:
                    break
                else:
                    true_history.insert(0, (old_query, old_response))
                    base_prompt_token_num += history_token_num

        return true_history

    def build_prompt(self, query, history=None):
        if history is None:
            history = []
        prompt = ""
        for i, (old_query, response) in enumerate(history):
            prompt += "[Round {}]\n\n问：{}\n\n答：{}\n\n".format(i + 1, old_query, response)
        prompt += "[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)
        return prompt

    def build_inputs(self, query: str, history: List[Tuple[str, str]] = None):
        prompt = self.build_prompt(query, history=history)
        inputs = self.tokenizer([prompt], return_tensors="pt")
        inputs = inputs.to(self.device)
        return inputs

    def lets_batch_chat(self, **kwargs):
        pass

    def lets_chat(self, prompt, history, stream=True, max_prompt_length=None, max_length=None, **kwargs):

        if max_length is None or max_length > self.max_length:
            max_length = self.max_length
        if max_prompt_length is None or max_prompt_length > self.max_prompt_length:
            max_prompt_length = self.max_prompt_length

        if self.logger:
            self.logger.info(
                str({'max_length': max_length, 'max_prompt_length': max_prompt_length}) + '\n' + str(kwargs) + '\n')

        history = self.select_history(prompt, history, max_prompt_length)

        input_prompt = self.build_prompt(prompt, history)
        prompt_tokens = self.token_counter(input_prompt)

        if self.logger:
            self.logger.info(str({'prompt_tokens': prompt_tokens, 'prompt_str_len': len(input_prompt),
                                  'prompt': input_prompt}) + '\n')

        if stream:
            def stream_generator():
                for resp in self.model.stream_chat(self.tokenizer, prompt, history, max_length=max_length, **kwargs):
                    generation_tokens = self.token_counter(resp[0])
                    torch_gc(self.device)
                    his = [list(x) for x in resp[1]]
                    yield {"answer": resp[0], "history": his,
                           "usage": {"prompt_tokens": prompt_tokens, "generation_tokens": generation_tokens,
                                     "total_tokens": prompt_tokens + generation_tokens}}

            return stream_generator()
        else:
            answer, history = self.model.chat(self.tokenizer, prompt, history, max_length=max_length, **kwargs)
            generation_tokens = self.token_counter(answer)

            torch_gc(self.device)
            his = [list(x) for x in history]

            return {"answer": answer, "history": his,
                    "usage": {"prompt_tokens": prompt_tokens, "generation_tokens": generation_tokens,
                              "total_tokens": prompt_tokens + generation_tokens}}
