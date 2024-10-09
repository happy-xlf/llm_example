#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   LLM.py
@Time    :   2024/05/17 14:53:47
@Author  :   Lifeng
@Version :   1.0
@Desc    :   None
'''

from typing import List, Dict, Optional, Tuple, Union
import copy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

class BaseModel:
    def __init__(self, path: str = ""):
        self.path =path
    
    def chat(self, prompt: str, history: List[dict]):
        pass

    def load_model(self):
        pass

class ChatGLM3Chat(BaseModel):
    def __init__(self, path: str = '') -> None:
        super().__init__(path)
        self.device = "cuda"
        self.load_model()

    def load_model(self):
        print('================ Loading model ================')
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.path, torch_dtype=torch.float16, trust_remote_code=True).cuda(self.device).eval()
        print('================ Model loaded ================')

    def chat(self, prompt: str, history: List[dict], meta_instruction:str ='') -> str:
        if history is None:
            history = []
        else:
            history = copy.deepcopy(history)
                
        messages = history_to_messages(history)
        if meta_instruction!="":
            messages.append({'role': 'system', 'content': meta_instruction})
        messages.append({'role': 'user', 'content': prompt})
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(model_inputs.input_ids, max_new_tokens=512, pad_token_id=self.tokenizer.eos_token_id)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        history.append((prompt, response))
        return response, history
    
# You can find it in Qwen1.5 chat demo
def history_to_messages(history):
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'}]
    for h in history:
        messages.append({'role': 'user', 'content': h[0]})
        messages.append({'role': 'assistant', 'content': h[1]})
    return messages

if __name__ == '__main__':
    model = ChatGLM3Chat(path="/home/lifeng/LLM_work/Models/ZhipuAI/chatglm3-6b")
    his = [{'role': 'user', 'content': '你好'}, {'role': 'assistant', 'metadata': '', 'content': '你好👋！我是人工智能助手 ChatGLM3-6B，很高兴见到你，欢迎问我任何问题。'}]
    print(model.chat("请问99+100=？", his, "你是一个只能机器人"))

