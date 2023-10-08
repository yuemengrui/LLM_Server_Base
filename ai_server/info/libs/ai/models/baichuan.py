# *_*coding:utf-8 *_*
# @Author : YueMengRui
import time
import torch
import numpy as np
from typing import List
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
        self.max_length = self.model.config.model_max_length
        self.max_prompt_length = self.max_length - self.model.generation_config.max_new_tokens

        if self.logger:
            self.logger.info(str({'config': self.model.config}) + '\n')
            self.logger.info(str({'config': self.model.generation_config}) + '\n')
            self.logger.info(str({'max_length': self.max_length, 'max_prompt_length': self.max_prompt_length}) + '\n')

        # warmup
        self.lets_chat('你好', [], stream=False)

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

    def check_token_len(self, prompt: str):
        code = True
        messages = [{'role': 'user', 'content': prompt}]
        prompt_token_len = self.token_counter(messages)
        if prompt_token_len > self.max_length:
            code = False

        return code, prompt_token_len, self.max_length

    def token_counter(self, messages: List):
        total_input, round_input = [], []
        for message in messages:
            content_tokens = self.tokenizer.encode(message['content'])
            if message['role'] == 'user':
                round_input = [self.model.generation_config.user_token_id] + content_tokens
            elif message['role'] == 'assistant':
                round_input = [
                                  self.model.generation_config.assistant_token_id
                              ] + content_tokens + [
                                  self.model.generation_config.eos_token_id
                              ]
            total_input = total_input + round_input

        return len(total_input)

    def select_history(self, prompt, history, max_prompt_length):
        base_prompt_token_num = self.token_counter([{'role': 'user', 'content': prompt}])
        true_history = []
        if history and base_prompt_token_num < max_prompt_length:
            for (old_query, old_response) in history[::-1]:
                history_token_num = self.token_counter(
                    [{'role': 'user', 'content': old_query}, {'role': 'assistant', 'content': old_response}])

                if base_prompt_token_num + history_token_num > max_prompt_length:
                    break
                else:
                    true_history.insert(0, [old_query, old_response])
                    base_prompt_token_num += history_token_num

        return true_history

    def _build_chat_input(self, messages):
        total_input, round_input = [], []
        for i, message in enumerate(messages[::-1]):
            content_tokens = self.tokenizer.encode(message['content'])
            if message['role'] == 'user':
                round_input = [self.model.generation_config.user_token_id] + content_tokens + round_input
                total_input = round_input + total_input
                round_input = []
            elif message['role'] == 'assistant':
                round_input = [
                                  self.model.generation_config.assistant_token_id
                              ] + content_tokens + [
                                  self.model.generation_config.eos_token_id
                              ] + round_input
            else:
                self.logger.warning(f"message role not supported yet: {message['role']}\n")
        if self.logger:
            self.logger.info(str({'prompt_len': len(total_input), 'prompt': self.tokenizer.decode(total_input)}) + '\n')
        total_input.append(self.model.generation_config.assistant_token_id)
        return total_input

    def lets_chat(self, prompt, history, stream, max_prompt_length=None, **kwargs):

        if max_prompt_length is None or max_prompt_length > self.max_prompt_length:
            max_prompt_length = self.max_prompt_length
        if self.logger:
            self.logger.info(str({'max_prompt_length': max_prompt_length}) + '\n' + str(kwargs) + '\n')

        history = self.select_history(prompt, history, max_prompt_length)

        messages = []
        for his in history:
            messages.append({'role': 'user', 'content': his[0]})
            messages.append({'role': 'assistant', 'content': his[1]})

        messages.append({'role': 'user', 'content': prompt})

        input_prompt = self._build_chat_input(messages)
        prompt_tokens = len(input_prompt)
        input_prompt_str = self.tokenizer.decode(input_prompt)
        if self.logger:
            self.logger.info(str({'prompt_tokens': prompt_tokens, 'prompt_str_len': len(input_prompt_str),
                                  'prompt': input_prompt_str}) + '\n')

        if stream:
            # history.append([prompt, ""])

            def stream_generator():
                start = time.time()
                for resp in self.model.chat(self.tokenizer, messages, stream=True, **kwargs):
                    generation_tokens = len(self.tokenizer.encode(resp))
                    time_cost = time.time() - start
                    average_speed = f"{generation_tokens / time_cost:.3f} token/s"
                    # history[-1][1] = resp
                    torch_gc(self.device)
                    yield {"answer": resp, "history": history, "time_cost": f"{time_cost:.3f}s",
                           "usage": {"prompt_tokens": prompt_tokens, "generation_tokens": generation_tokens,
                                     "total_tokens": prompt_tokens + generation_tokens, "average_speed": average_speed}}

            return stream_generator()

        else:
            start = time.time()
            resp = self.model.chat(self.tokenizer, messages, **kwargs)
            generation_tokens = len(self.tokenizer.encode(resp))
            time_cost = time.time() - start
            average_speed = f"{generation_tokens / time_cost:.3f} token/s"
            # history.append([prompt, resp])

            torch_gc(self.device)

            return {"answer": resp, "history": history, "time_cost": f"{time_cost:.3f}s",
                    "usage": {"prompt_tokens": prompt_tokens, "generation_tokens": generation_tokens,
                              "total_tokens": prompt_tokens + generation_tokens, "average_speed": average_speed}}

    def lets_batch_chat(self, prompt_list, history_list, stream, max_prompt_length=None, **kwargs):
        if max_prompt_length is None or max_prompt_length > self.max_prompt_length:
            max_prompt_length = self.max_prompt_length
        if self.logger:
            self.logger.info(str({'max_prompt_length': max_prompt_length}) + '\n' + str(kwargs) + '\n')

        batch_inputs = []
        batch_prompt_tokens = []
        batch_history_list = []
        for i in range(len(prompt_list)):
            history = self.select_history(prompt_list[i], history_list[i], max_prompt_length)
            batch_history_list.append(history)
            batch_history_list[-1].append([prompt_list[i], ""])

            messages = []
            for his in history:
                messages.append({'role': 'user', 'content': his[0]})
                messages.append({'role': 'assistant', 'content': his[1]})

            messages.append({'role': 'user', 'content': prompt_list[i]})

            input_prompt = self._build_chat_input(messages)
            prompt_tokens = len(input_prompt)
            input_prompt_str = self.tokenizer.decode(input_prompt)
            if self.logger:
                self.logger.info(str({'prompt_tokens': prompt_tokens, 'prompt_str_len': len(input_prompt_str),
                                      'prompt': input_prompt_str}) + '\n')

            batch_prompt_tokens.append(prompt_tokens)
            batch_inputs.append(np.array(input_prompt))

        max_length = max([len(x) for x in batch_inputs])
        # left padding
        batch_inputs = np.array([np.pad(t, (max_length - t.shape[0], 0), 'constant',
                                        constant_values=self.model.generation_config.pad_token_id) for t in
                                 batch_inputs])
        # right padding
        # batch_inputs = np.array([np.pad(t, (0, max_length - t.shape[0]), 'constant',
        #                                 constant_values=self.model.generation_config.pad_token_id) for t in
        #                          batch_inputs])

        batch_inputs = torch.LongTensor(batch_inputs).to(self.device)
        batch_len = len(prompt_list)
        if stream:

            def stream_generator():
                start = time.time()
                for resp_list in self.model.batch_chat(self.tokenizer, batch_inputs, self.model.generation_config,
                                                       stream=True, **kwargs):
                    outputs = []
                    for i in range(batch_len):
                        generation_tokens = len(self.tokenizer.encode(resp_list[i]))
                        average_speed = f"{generation_tokens / (time.time() - start):.3f} token/s"
                        batch_history_list[i][-1][-1] = resp_list[i]

                        outputs.append({"answer": resp_list[i], "history": batch_history_list[i],
                                        "usage": {"prompt_tokens": batch_prompt_tokens[i],
                                                  "generation_tokens": generation_tokens,
                                                  "total_tokens": batch_prompt_tokens[i] + generation_tokens,
                                                  "average_speed": average_speed}})

                    torch_gc(self.device)
                    yield outputs

            return stream_generator()

        else:
            start = time.time()
            resp_list = self.model.batch_chat(self.tokenizer, batch_inputs, self.model.generation_config, **kwargs)
            outputs = []
            for i in range(batch_len):
                generation_tokens = len(self.tokenizer.encode(resp_list[i]))
                average_speed = f"{generation_tokens / (time.time() - start):.3f} token/s"
                batch_history_list[i][-1][-1] = resp_list[i]

                outputs.append({"answer": resp_list[i], "history": batch_history_list[i],
                                "usage": {"prompt_tokens": batch_prompt_tokens[i],
                                          "generation_tokens": generation_tokens,
                                          "total_tokens": batch_prompt_tokens[i] + generation_tokens,
                                          "average_speed": average_speed}})

            torch_gc(self.device)

            return outputs
