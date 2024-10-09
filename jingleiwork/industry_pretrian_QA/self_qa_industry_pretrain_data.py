from transformers import AutoTokenizer, AutoModel
import os 
import json
import re
import torch
import time

# 指定gpu显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 清空GPU缓存
torch.cuda.empty_cache()

SYSTEM_PROMPT = """
你是一个智能问答生成机器人，专注于根据提供的文本段落生成相关的问答对。请确保：
1. 每个问答对严格包含一个指令（instruction）、一个输入信息（input）和一个输出信息（output）。指令定义了需要解决的问题或要执行的任务，输入信息提供了执行指令所需的具体数据或情景描述，输出信息则是根据指令和输入生成的具体答案。
2. 输出的问答对需多样化、高质量，并且紧密地与提供的指令和上下文信息相关。确保每个输出直接回应输入的问题，避免生成多个输出。
3. 所有问答对尽量使用简体中文完成，语言表达需要准确、清晰、流畅。
4. 避免生成重复的问答对，每个问答对都应当是独特的，能够提供独到的见解或信息。
5. 输出必须遵守严格的JSON格式，其中“instruction”字段表示任务，"input"字段提供具体情境，"output"字段给出解决方案，有且只有这三个字段，三个字段缺一不可。

示例格式如下：
[
    {
        "instruction": "描述如何查询火车票信息。",
        "input": "我需要从北京到上海的火车票信息。",
        "output": "你可以通过12306网站或其手机应用程序查询和购买从北京到上海的火车票。"
    },
    {
        "instruction": "解释如何进行在线退票。",
        "input": "我昨天购买了一张去杭州的火车票，今天需要取消。",
        "output": "你可以在购票平台上找到你的订单并选择退票选项进行操作，或者使用12306的客户端来处理退票。"
    }
]

你的目标是根据从文本文件中分割得到的段落，生成准确的、与指令紧密相连的问答对。请始终确保每个问答对的信息完整且准确地反映了输入的指令和上下文。

给定文本内容如下：
"""


model_dir = "/home/lifeng/LLM_work/Models/ZhipuAI/chatglm3-6b"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).half().cuda()
model = model.eval()

# 中文文本分段处理
def split_text(text, maxlength=256):
    # 根据标点符号分段，确保每段文本长度不超过最大长度限制
    sentences = re.split('([。！？.])', text)
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
            if current_segment:
                segments.append(current_segment)
                current_segment = ''
            # 当单个句子长度超过最大长度时，进一步分割该句子
            while len(sentence) > maxlength:
                segments.append(sentence[:maxlength])
                sentence = sentence[maxlength:]
            # 将剩余的句子添加到当前片段中
            current_segment = sentence

    if current_segment:
        segments.append(current_segment)

    return segments


# 处理单个文件
def process_file(file_path):
    try: 
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        segments = split_text(text)
        print(f"文件 {file_path} 分割完成，共 {len(segments)} 个片段。")
        qa_pairs = []
        history = []
        for segment in segments:
            try:
                prompt = SYSTEM_PROMPT + f"{segment} 请根据上述信息和示例生成具体相关的QA对，确保信息的实用性和指导性。"
                qa_text, history = model.chat(tokenizer, prompt, history=[])
                # print(f"QA text: \n{qa_text}")    
                qa_data = json.loads(qa_text)
                qa_pairs.extend(qa_data)
        
            except json.JSONDecodeError:
                print(f"解析json时出错")
                qa_data = reconstruct_json(qa_text)
                qa_pairs.extend(qa_data)
            except Exception as e_inner:
                print(f"生成QA对时出错: {e_inner}")

                continue

    except Exception as e_outer:
        print(f"处理文件 {file_path} 时出错: {e_outer}")

    return qa_pairs

# 重构json格式
def reconstruct_json(qa_text): 
    qa_list = []
    if qa_text is None or qa_text == "":
        return qa_list
    pattern = r'"instruction":\s*"([^"]*)",\s*"input":\s*"([^"]*)",\s*"output":\s*"([^"]*)"'
    matches = re.finditer(pattern, qa_text)
    for match in matches:
        try:
            instruction, input, output = match.groups()
            qa_dict = {
                "instruction": instruction,
                "input": input,
                "output": output
            }
            qa_list.append(qa_dict)
        except Exception as e:
            print(f"重构JSON时出错: {e}")
            print(f"mathch: {match.groups()}")
            print(f"qa_text")
            continue

    return qa_list


def main():
    start_time = time.time()
    script_dir = os.path.dirname(__file__)
    directory = "/home/lifeng/jinglei_work/data/industry_pretrain_data"
    for filename in os.listdir(directory):
        print(f"Processing file: {filename}")
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            qa_pairs = process_file(file_path)
            base_name = filename[:-4].replace(" ","").replace("-","_")
            qa_path = os.path.join(script_dir, f"QA_{base_name}.json")
            with open(qa_path, "w", encoding="utf-8") as f:
                json.dump(qa_pairs, f, ensure_ascii=False, indent=4)
            print(f"{len(qa_pairs)} QA pairs saved to QA_{base_name}.json, time:{time.time() - start_time:.2f} seconds.")
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()