#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : qwen_agent.py
# @Time      : 2024/05/13 22:14:18
# @Author    : lifeng


import os
import json
from langchain_community.tools.tavily_search import TavilySearchResults
import datetime
from typing import List, Dict, Optional, Tuple, Union
import copy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

device='cuda' if torch.cuda.is_available() else 'cpu'
path = "/home/lifeng/LLM_work/Models/qwen/Qwen1___5-7B-Chat"
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, trust_remote_code=True).cuda(device).eval()


def llm(query,history=[],user_stop_words=[]):    # 调用api_server
    try:
        messages=[{'role':'system','content':'You are a helpful assistant.'}]
        for hist in history:
            messages.append({'role':'user','content':hist[0]})
            messages.append({'role':'assistant','content':hist[1]})
        messages.append({'role':'user','content':query})
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        if user_stop_words[0] in response:
            response=response.split(user_stop_words[0])[0]
        return response
    except Exception as e:
        return str(e)
    
# travily搜索引擎
# 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
# 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
os.environ['TAVILY_API_KEY']='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'    # travily搜索引擎api key
tavily=TavilySearchResults(max_results=5)
tavily.description='这是一个类似谷歌和百度的搜索引擎，搜索知识、天气、股票、电影、小说、百科等都是支持的哦，如果你不确定就应该搜索一下，谢谢！'

# 工具列表
tools=[tavily, ]

tool_names='or'.join([tool.name for tool in tools])  # 拼接工具名
tool_descs=[] # 拼接工具详情
for t in tools:
    args_desc=[]
    for name,info in t.args.items():
        args_desc.append({'name':name,'description':info['description'] if 'description' in info else '','type':info['type']})
    args_desc=json.dumps(args_desc,ensure_ascii=False)
    tool_descs.append('%s: %s,args: %s'%(t.name,t.description,args_desc))
tool_descs='\n'.join(tool_descs)

prompt_tpl='''Today is {today}. Please Answer the following questions as best you can. You have access to the following tools:

{tool_descs}

These are chat history before:
{chat_history}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {query}
{agent_scratchpad}
'''

def agent_execute(query,chat_history=[]):
    global tools,tool_names,tool_descs,prompt_tpl,llm,tokenizer
    
    agent_scratchpad='' # agent执行过程
    while True:
        # 1）触发llm思考下一步action
        history='\n'.join(['Question:%s\nAnswer:%s'%(his[0],his[1]) for his in chat_history])
        today=datetime.datetime.now().strftime('%Y-%m-%d')
        prompt=prompt_tpl.format(today=today,chat_history=history,tool_descs=tool_descs,tool_names=tool_names,query=query,agent_scratchpad=agent_scratchpad)
        print('\033[32m---等待LLM返回... ...\n%s\n\033[0m'%prompt,flush=True)
        response=llm(prompt,user_stop_words=['Observation:'])
        print('\033[34m---LLM返回---\n%s\n---\033[34m'%response,flush=True)
        
        # 2）解析thought+action+action input+observation or thought+final answer
        thought_i=response.rfind('Thought:')
        final_answer_i=response.rfind('\nFinal Answer:')
        action_i=response.rfind('\nAction:')
        action_input_i=response.rfind('\nAction Input:')
        observation_i=response.rfind('\nObservation:')
        
        # 3）返回final answer，执行完成
        if final_answer_i!=-1 and thought_i<final_answer_i:
            final_answer=response[final_answer_i+len('\nFinal Answer:'):].strip()
            chat_history.append((query,final_answer))
            return True,final_answer,chat_history
        
        # 4）解析action
        if not (thought_i<action_i<action_input_i):
            return False,'LLM回复格式异常',chat_history
        if observation_i==-1:
            observation_i=len(response)
            response=response+'Observation: '
        thought=response[thought_i+len('Thought:'):action_i].strip()
        action=response[action_i+len('\nAction:'):action_input_i].strip()
        action_input=response[action_input_i+len('\nAction Input:'):observation_i].strip()
        
        # 5）匹配tool
        the_tool=None
        for t in tools:
            if t.name==action:
                the_tool=t
                break
        if the_tool is None:
            observation='the tool not exist'
            agent_scratchpad=agent_scratchpad+response+observation+'\n'
            continue 
        
        # 6）执行tool
        try:
            action_input=json.loads(action_input)
            tool_ret=the_tool.invoke(input=action_input)

            from tavily import TavilyClient
            tavily = TavilyClient(api_key="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")
            # For basic search:
            ret = tavily.search(query="北京 今日 天气")
            context = [obj["content"] for obj in ret['results']]
            if tool_ret == []:
                tool_ret = context

        except Exception as e:
            observation='the tool has error:{}'.format(e)
        else:
            observation=str(tool_ret)
        agent_scratchpad=agent_scratchpad+response+observation+'\n'

def agent_execute_with_retry(query,chat_history=[],retry_times=3):
    for i in range(retry_times):
        success,result,chat_history=agent_execute(query,chat_history=chat_history)
        if success:
            return success,result,chat_history
    return success,result,chat_history

my_history=[]
while True:
    query=input('query:')
    success,result,my_history=agent_execute_with_retry(query,chat_history=my_history)
    my_history=my_history[-10:]