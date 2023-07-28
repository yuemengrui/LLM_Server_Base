# *_*coding:utf-8 *_*
# @Author : YueMengRui
import json
import torch
import numpy as np
import torch.nn.functional as F
from .base_model import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig


def torch_gc(device):
    if torch.cuda.is_available():
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


class BaiChuan(BaseModel):

    def __init__(self, model_name_or_path, logger=None, device='cuda', **kwargs):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.logger = logger
        self._load_model(model_name_or_path, device)
        if self.logger:
            self.logger.info(str({'config': self.model.config}) + '\n')
            self.logger.info(str({'config': self.model.generation_config}) + '\n')

    def _load_model(self, model_name_or_path, device):

        if device == 'mps':
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=True
            ).half().to('mps')
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        self.model.generation_config = GenerationConfig.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=False,
            trust_remote_code=True
        )
        self.device = self.model.device

    def get_embeddings(self, sentences):
        embeddings = []
        for text in sentences:
            input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
            model_output = self.model(input_ids, output_hidden_states=True)
            data = model_output.hidden_states[-1][0]
            data = F.normalize(torch.mean(data, dim=0), p=2, dim=0)

            embeddings.append(data)

        return embeddings

    def _build_chat_input(self, messages, max_new_tokens=0):
        max_new_tokens = max_new_tokens or self.model.generation_config.max_new_tokens
        max_input_tokens = self.model.config.model_max_length - max_new_tokens
        max_input_tokens = max(self.model.config.model_max_length // 2, max_input_tokens)
        self.logger.info(str({'max_input_tokens': max_input_tokens}) + '\n')
        total_input, round_input = [], []
        for i, message in enumerate(messages[::-1]):
            content_tokens = self.tokenizer.encode(message['content'])
            if message['role'] == 'user':
                round_input = [self.model.generation_config.user_token_id] + content_tokens + round_input
                if total_input and len(total_input) + len(round_input) > max_input_tokens:
                    break
                else:
                    total_input = round_input + total_input
                    if len(total_input) >= max_input_tokens:
                        break
                    else:
                        round_input = []
            elif message['role'] == 'assistant':
                round_input = [
                                  self.model.generation_config.assistant_token_id
                              ] + content_tokens + [
                                  self.model.generation_config.eos_token_id
                              ] + round_input
            else:
                self.logger.warning(f"message role not supported yet: {message['role']}\n")
        total_input = total_input[-max_input_tokens:]  # truncate left
        if self.logger:
            self.logger.info(str({'prompt_len': len(total_input), 'prompt': self.tokenizer.decode(total_input)}) + '\n')
        total_input.append(self.model.generation_config.assistant_token_id)
        return total_input

    def letschat(self, query_list, history_list, **kwargs):
        if self.logger:
            self.logger.info(str(kwargs) + '\n')
        batch_prompt = []

        for i in range(len(query_list)):
            query = query_list[i]
            history = history_list[i]
            messages = []
            for his in history:
                messages.append({'role': 'user', 'content': his[0]})
                messages.append({'role': 'assistant', 'content': his[1]})

            messages.append({'role': 'user', 'content': query})

            batch_prompt.append(np.array(self._build_chat_input(messages)))

        max_length = max([len(x) for x in batch_prompt])
        # left padding
        batch_prompt = np.array([np.pad(t, (max_length - t.shape[0], 0), 'constant',
                                        constant_values=self.model.generation_config.pad_token_id) for t in
                                 batch_prompt])
        # right padding
        # batch_prompt = np.array([np.pad(t, (0, max_length - t.shape[0]), 'constant',
        #                                 constant_values=self.model.generation_config.pad_token_id) for t in
        #                          batch_prompt])

        batch_input = torch.LongTensor(batch_prompt).to(self.device)

        for ind in range(len(batch_prompt)):
            history_list[ind].append(['', ''])

        self.model.generation_config.update(**kwargs)

        resp_list = self.model.batch_chat(self.tokenizer, batch_input, self.model.generation_config)

        torch_gc(self.device)

        return resp_list

    def lets_stream_chat(self, query_list, history_list, **kwargs):
        torch_gc(self.device)
        if self.logger:
            self.logger.info(str(kwargs) + '\n')
        batch_prompt = []

        for i in range(len(query_list)):
            query = query_list[i]
            history = history_list[i]
            messages = []
            for his in history:
                messages.append({'role': 'user', 'content': his[0]})
                messages.append({'role': 'assistant', 'content': his[1]})

            messages.append({'role': 'user', 'content': query})

            batch_prompt.append(np.array(self._build_chat_input(messages)))

        max_length = max([len(x) for x in batch_prompt])
        # left padding
        batch_prompt = np.array([np.pad(t, (max_length - t.shape[0], 0), 'constant',
                                        constant_values=self.model.generation_config.pad_token_id) for t in
                                 batch_prompt])
        # right padding
        # batch_prompt = np.array([np.pad(t, (0, max_length - t.shape[0]), 'constant',
        #                                 constant_values=self.model.generation_config.pad_token_id) for t in
        #                          batch_prompt])

        batch_input = torch.LongTensor(batch_prompt).to(self.device)

        for ind in range(len(batch_prompt)):
            history_list[ind].append(['', ''])

        generation_config = self.model.generation_config.update(**kwargs)
        for response in self.model.batch_stream_chat(self.tokenizer, batch_input, generation_config):
            yield response, history_list
