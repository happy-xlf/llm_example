from newspaper import Article
from transformers import AutoTokenizer, AutoModel
import os 
import json
import re

# 指定gpu显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

SYSTEM_PROMPT = """
    你是一个能根据提供的文本内容生成QA对的机器人。以下是你的任务要求：
    1. 生成尽可能多的QA对。
    2. 每个QA对包含一个问题和一个简洁的答案。
    3. 答案必须用简体中文。
    4. 生成的QA对不能重复。
    5. 使用json格式将QA对包裹起来，问题用"question"表示，答案用"answer"表示。
    
    示例格式：
    [
        {
            "question": "...",
            "answer": "..."
        },
        {
            "question": "...",
            "answer": "..."
        }
    ]
    以下是给定的文本内容：
    """

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

# 中文文本分段处理
def split_text(text, maxlength=512):
    # 根据中文标点符号分段，确保每段文本长度不超过最大长度限制
    sentences = re.split('([。])', text)
    segments = []
    current_segment = ""
    for i in range(0, len(sentences) - 1, 2):
        # sentences[i] 表示列表中的句子部分
        # sentences[i+1] 表示随后的标点符号部分
        # 如果句子是列表中的最后一个元素，它后面可能没有标点符号
        sentence = sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else "")
        if len(current_segment) + len(sentence) <= maxlength:
            current_segment += sentence
        else:
            segments.append(current_segment)
            current_segment = sentence

    if current_segment:
        segments.append(current_segment)

    return segments
        

def main():
    
    url = "https://zhuanlan.zhihu.com/p/638426349"
    content = get_article_text(url)
    if content != "":
        segments = split_text(content)
        qa_pairs = []
        history = []
        
        # 生成QA对
        for segment in segments:
            prompt = SYSTEM_PROMPT + f"{segment} 请开始生成 QA 对:"
            qa_text, history = model.chat(tokenizer, prompt, history=history)
            qa_data = json.loads(qa_text)
            qa_pairs.extend(qa_data)
            
        print(qa_pairs)
        script_dir = os.path.dirname(__file__)
        file_path = os.path.join(script_dir, "QA_extension.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=4)
        print("QA对已保存至QA_extension.json文件")
    else:
        print("获取文章内容失败")


if __name__ == '__main__':
    main()