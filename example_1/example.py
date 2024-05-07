#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   self_qa_example.py
@Time    :   2024/04/30 10:36:11
@Author  :   Lifeng
@Version :   1.0
@Desc    :   None
'''
#这是一个用于从网页中提取文章的库。Article类可以帮助我们抓取网页内容、标题、作者等信息。
from newspaper import Article
#这是一个用于文本生成模型的库。AutoTokenizer用于对文本进行分词、转换为索引等功能，AutoModel用于训练和生成文本。
from transformers import AutoTokenizer, AutoModel
#分别用于处理文件和数据交换格式。
import os,json

# 指定gpu显卡(显卡1)
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

#这段代码主要用于加载一个名为chatglm3-6b的预训练模型，并将其准备用于生成对话。通过使用半精度浮点格式和将模型移动到CUDA上，可以提高模型的性能。
model_dir = "/home/lifeng/LLM_work/Models/ZhipuAI/chatglm3-6b"
#AutoTokenizer.from_pretrained：这是一个函数，它从给定的模型目录中加载预训练的Tokenizer。Tokenizer负责将输入文本转换为模型可接受的输入格式。在这个例子中，我们从模型目录中加载名为chatglm3-6b的预训练Tokenizer。
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
#AutoModel.from_pretrained：这是一个函数，它从给定的模型目录中加载预训练的模型。在这个例子中，我们从模型目录中加载名为chatglm3-6b的预训练模型。
#half()：这是一个方法，它将模型转换为半精度浮点格式（即半精度FP16），以提高模型的性能。在这个例子中，我们将其转换为半精度浮点格式。
#cuda()：这是一个方法，它将模型移动到CUDA（应用程序矩形器单元）上。这对于在支持CUDA的设备上运行模型非常有用，例如在GPU上运行模型。在这个例子中，我们将其移动到CUDA上。
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).half().cuda()
#model.eval()：这是一个方法，它将模型设置为评估模式（evaluation mode）。在评估模式下，模型将不会包含任何可训练的参数，从而提高模型的性能。在这个例子中，我们将模型设置为评估模式。
model = model.eval()

# 网页文章解析
#函数内部，首先使用Article类的构造函数创建一个Article对象（变量a）
#然后尝试下载文章内容download()并解析parse()。如果解析成功，函数将返回解析后的文本内容return a.text；
# 如果解析失败，函数将捕获异常Exception as e:并打印错误信息print(f"url解析失败，错误原因：{e}")，同时返回一个空字符串。
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
    if content!="":
        
        query = """
        {content}

        请开始生成 QA 对:
        """.format(content = content)

#将用户输入的query转换为系统回复，并打印出回复。这个聊天机器人可以用于与用户进行文字交流，通过语音和文本的方式与用户进行交互。
        SYSTEM_PROMPT += query
#参数说明：model：预训练的语言模型，如GPT-3.tokenizer：分词器，用于将输入文本分词并转换为模型可接受的输入格式SYSTEM_PROMPT：系统提示，通常是预先设置好的字符串，用于提示用户输入
#history：对话历史，是一个列表，包含之前的对话信息，用于模型进行训练和预测
        response, history = model.chat(tokenizer, SYSTEM_PROMPT, history=[])
        print(response)
#从一个名为response的变量中加载数据
        data = json.loads(response)
#这段代码是用于将一个名为data的Python对象保存到一个名为QA.json的文件中。json模块负责将data对象转换为JSON格式，并将其写入到QA.json文件中。
#实现原理： with语句用于确保文件在操作完成后被正确关闭。open()函数的第一个参数是文件名，第二个参数是文件打开模式（"w"表示写入模式），第三个参数是文件编码（encoding="utf-8"表示使用UTF-8编码）。
#json.dump()函数负责将data对象转换为JSON格式，并将其写入到文件中。ensure_ascii=False表示在输出中允许使用非ASCII字符，indent=4表示在JSON输出中使用4个空格进行缩进。       
        with open("./QA.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    else:
        print("获取文章内容失败")





