#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   tool.py
@Time    :   2024/05/17 15:13:44
@Author  :   Lifeng
@Version :   1.0
@Desc    :   None
'''

import os, json, re
import requests

"""
工具函数

- 首先要在 tools 中添加工具的描述信息
- 然后在 tools 中添加工具的具体实现

- https://serper.dev/dashboard
"""

class Tools:
    def __init__(self) -> None:
        self.toolConfig = self._tools()
    
    def _tools(self):
        tools = [
            {
                'name_for_human': '谷歌搜索',
                'name_for_model': 'google_search',
                'description_for_model': '谷歌搜索是一个通用搜索引擎，可用于访问互联网、查询百科知识、了解时事新闻等。',
                'parameters': [
                    {
                        'name': 'query',
                        'description': '搜索关键词或短语',
                        'required': True,
                        'schema': {'type': 'string'},
                    }
                ],
            },
            {
                'name_for_human': '计算器',
                'name_for_model': 'calculator',
                'description_for_model': '用于数学计算，支持加减乘除等基本运算。',
                'parameters': [
                    {
                        'name': 'num',
                        'description': '只接受纯数字，小数，整数',
                        'required': True,
                        'schema': {'type': 'string'},
                    }
                ],
            }
        ]
        return tools

    def google_search(self, query: str):
        url = "https://google.serper.dev/search"

        payload = json.dumps({"q": query})
        headers = {
            'X-API-KEY': 'bd55923dd39d2388ffeb1da866df03d82b91ff07',
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload).json()

        return response['organic'][0]['snippet']
    
    def calculator(self, num):
        # 用re找到字符串里的数字包含小数
        search_num = re.search(r'\d+\.?\d*', num)
        num = float(search_num)
        return num * 2
    
# if __name__ == '__main__':
#     tool = Tools()
#     print(tool.google_search('刘关张桃园三结义都是谁？'))
