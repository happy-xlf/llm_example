#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   self_qa_example.py
@Time    :   2024/04/30 10:36:11
@Author  :   Lifeng
@Version :   1.0
@Desc    :   None
'''

from newspaper import Article
from transformers import AutoTokenizer, AutoModel
import os,json

# 指定gpu显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

SYSTEM_PROMPT = """
    你是一名QA对生成机器人，你会根据我提供的【文本内容】自动生成合适的QA对，要求如下：

    对于我给的文本内容，你需要生成五条这样的QA对
    QA对内容不能重复，答案不能过长
    用简体中文回答
    生成的 QA 对需要用 json 代码块包裹起来,Q请用question表示，A请用answer表示
    例如：
    [
        {
            "question": "...",
            "answer": "..."
        },
        {
            "question": "...",
            "answer": "..."
        },
        ...
    ]
    
    以下是给定的文本内容："""

model_dir = "/home/lifeng/LLM_work/Models/ZhipuAI/chatglm3-6b"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).half().cuda()
model = model.eval()

# 网页文章解析
def get_article_text(url):
    a = Article(url)
    try:
        a.download()
        a.parse()
        return a.text
    except Exception as e:
        print(f"url解析失败，错误原因：{e}")
        return ""


if __name__ == '__main__':
    
    url = "https://zhuanlan.zhihu.com/p/638426349"

    content = get_article_text(url)
    if content != "":
        
        query = """
        {content}

        请开始生成 QA 对:
        """.format(content = content)

        SYSTEM_PROMPT += query
        response, history = model.chat(tokenizer, SYSTEM_PROMPT, history=[])
        print(response)

        data = json.loads(response)
        with open("./QA.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    else:
        print("获取文章内容失败")





