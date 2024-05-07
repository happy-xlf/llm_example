from newspaper import Article
from transformers import AutoTokenizer, AutoModel
import os 
import json

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

def main():

    url = "https://zhuanlan.zhihu.com/p/638426349"
    content = get_article_text(url)
    if content != "":
        history = []
        prompt = SYSTEM_PROMPT + content
        response, history = model.chat(tokenizer, prompt, history=history)
        print(response)
        data = json.loads(response)
        print(data)

        try:
            script_dir = os.path.dirname(__file__)
            file_path = os.path.join(script_dir, "QA.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print("QA对已保存至QA.json文件")
        except Exception as e:
            print(f"保存QA对失败，错误原因：{e}")
    else:
        print("获取文章内容失败")


if __name__ == '__main__':
    main()