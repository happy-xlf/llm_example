#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   test.PY
@Time    :   2024/05/17 18:40:33
@Author  :   Lifeng
@Version :   1.0
@Desc    :   None
'''

from tavily import TavilyClient

tavily = TavilyClient(api_key="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")
# For basic search:
response = tavily.search(query="北京 今日 天气")
print(response)
# For advanced search:
response = tavily.search(query="北京 今日 天气", search_depth="advanced")
print(response)
# Get the search results as context to pass an LLM:
context = [{"url": obj["url"], "content": obj["content"]} for obj in response['results']]

print(context)

