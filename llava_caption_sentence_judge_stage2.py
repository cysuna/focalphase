import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from itertools import islice

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from transformers import set_seed


from PIL import Image
import math
import numpy as np
import re
import jsonlines
def convert_jsonl_to_json(jsonl_file):
    data = []
    with jsonlines.open(jsonl_file) as reader:
        for obj in reader:
            obj['id']=obj['question_id']
            data.append(obj)
    return data
    


import spacy
import nltk
from nltk.stem import WordNetLemmatizer
nlp = spacy.load("en_core_web_lg")

def check_synonyms_word(word1, word2, similarity_score):
    token1 = nlp(word1)
    token2 = nlp(word2)
    similarity = token1.similarity(token2)
    return similarity > similarity_score


def extract_nouns(text):
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    nouns = [lemmatizer.lemmatize(word) for word, pos in tagged if pos.startswith('NN')]
    return nouns


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def batch(iterable, n=1):
    """Yield successive n-sized batches from iterable."""
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch

def get_first_k_sentences(text, k):
    # 使用正则表达式匹配分句（非贪婪匹配，直到遇到逗号或句号）
    sentences = re.findall(r'.*?[,.]+', text)
    # 去除每个分句的首尾空格，并过滤掉空字符串
    sentences = [s.strip() for s in sentences if s.strip()]
    # 取前k个分句
    selected_sentences = sentences[:k]
    # 组合成一个字符串
    result = ' '.join(selected_sentences)
    return result

def get_no_k_sentences(text, k):
    # 使用正则表达式匹配分句（非贪婪匹配，直到遇到逗号或句号）
    sentences = re.findall(r'.*?[,.]+', text)
    # 去除每个分句的首尾空格，并过滤掉空字符串
    sentences = [s.strip() for s in sentences if s.strip()]
    # 取前k个分句
    selected_sentences = sentences[k]
    # 组合成一个字符串
    # result = ' '.join(selected_sentences)
    return selected_sentences

def count_valid_sentences(text):
    # 使用正则表达式匹配分句（非贪婪匹配，直到遇到逗号或句号）
    sentences = re.findall(r'.*?[,.]+', text)
    # 去除每个分句的首尾空格，并过滤掉空字符串
    valid_sentences = [s.strip() for s in sentences if s.strip()]
    # 返回有效分句的数量
    return len(valid_sentences)

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    
    # 确保tokenizer有pad_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    try:
        questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
        questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    except:
        questions = json.load(open(args.question_file,'r'))
    
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    
    if args.part >= 0:
        start = len(questions)*(args.part)//8
        end = len(questions)*(args.part+1)//8
        questions = questions[start:end]  
    
    # 强制设置batch_size=1
    args.batch_size = 1
    no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]
    yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    with open(answers_file, "w") as ans_file:
        buffer = []  # 初始化缓冲区
        buffer_size = 1000  # 设置缓冲区大小    
        # 逐样本处理（不再使用批处理）
        for line in tqdm(questions, total=len(questions)):
            image_file = line["image"]
            seq = line["sentence"].replace("<s> ","")
            objects = line["objects"]
            if len(objects)==0:
                continue            
            if objects[0]=="s":
                objects = objects[1:]            
            if len(objects)<2:
                continue
            # qs = f"Given an image and a sentence: '{seq}', verify if the sentence matches the image by checking if all the following objects exist in the image: {objects}. Respond strictly with 'yes' only if the entire sentence is accurate and all objects are present, otherwise 'no'. Do not explain or add any text."
            qs = f"Given an image and a sentence: '{seq}', carefully verify if the sentence accurately describes the image by checking: 1) whether all specified objects ({objects}) are present, 2) their quantities match if mentioned, 3) their spatial relationships if described, and 4) any attributes mentioned (colors, sizes, etc.). Respond strictly with 'Yes' only if every aspect of the sentence perfectly matches the image, otherwise respond 'No'."
            cur_prompt = qs
            if model.config.mm_use_im_start_end:
                qstemp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qstemp = DEFAULT_IMAGE_TOKEN + '\n' + qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt() 
            # print(prompt)
            # quit()
            # 单样本处理，无需填充
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            
            # 处理图像
            image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')


            image_tensor = process_images([image], image_processor, model.config)[0].unsqueeze(0).half().cuda()
            
            # 生成文本（单样本，无需attention_mask）
            with torch.inference_mode():
                outputs = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[image.size],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=20,
                    use_cache=True,
                    amp_attn=True,
                    amp_attn_cfg=None,
                    return_dict_in_generate=True,  # This is important to get the scores
                    output_scores=True,             # This tells the model to return logits
                )
            
            # Get the generated token IDs
            output_ids = outputs.sequences
            
            # Get the logits for each generation step
            # This is a tuple of tensors, one for each generation step
            generation_logits = outputs.scores
            
            # Get the logits for the first generated token
            first_token_logits = generation_logits[0]  # Shape: [batch_size, vocab_size]
            
            # Convert logits to probabilities using softmax
            probs = torch.nn.functional.softmax(first_token_logits, dim=-1)
            
            # Get the token IDs for "no" and "yes"
            # Note: You might need to adjust these based on your specific tokenizer

            
            # Get the probabilities for "no" and "yes"
            no_prob = probs[0, no_token_id].item()
            yes_prob = probs[0, yes_token_id].item()
            
            # print(f"Probability of 'no': {no_prob:.4f}")
            # print(f"Probability of 'yes': {yes_prob:.4f}")
            output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # 保存结果
            ans_id = shortuuid.uuid()
            buffer.append(json.dumps({
                    "question_id": line["question_id"],
                    # "prompt": line["text"],
                    "image": line["image"],
                    "sentence":seq,
                    "objects":objects,
                    "no_prob":float(no_prob),
                    "yes_prob":float(yes_prob),
                    # "caption":line["text"],
                    "text": output.strip(),
                    "answer_id": ans_id,
                    "model_id": model_name,
            }) + "\n")
            # 当缓冲区达到指定大小时写入文件
            if len(buffer) >= buffer_size:
                ans_file.writelines(buffer)
                ans_file.flush()
                buffer = []  # 清空缓冲区
        
        # 处理剩余不足buffer_size的记录
        if buffer:
            ans_file.writelines(buffer)
            ans_file.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="model/llava-1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="dataset/coco/coco2017/images/train2017")
    parser.add_argument("--question-file", type=str, default="clip_ft_dataset/detail_23k_modi/detail_23k.json")
    parser.add_argument("--answers-file", type=str, default="LLaVA_original_0616/exp/123213.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--startid", type=int, default=0, help="startid")
    parser.add_argument("--endid", type=int, default=0, help="endid")
    parser.add_argument("--part", type=int, default=-1, help="endid")

    args = parser.parse_args()

    eval_model(args)
