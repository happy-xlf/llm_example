#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   agent_demo.py
@Time    :   2024/05/17 15:45:18
@Author  :   Lifeng
@Version :   1.0
@Desc    :   None
'''

from tiny_agent.Agent import Agent

agent = Agent('/home/lifeng/LLM_work/Models/qwen/Qwen1___5-7B-Chat')

print(agent.system_prompt)

text = input("请输入查询内容：")
while text!="q":
    response, _ = agent.text_completion(text=text, history=[])
    print(response)
    text = input("请输入查询内容：")