import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from itertools import islice


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

def add_gaussian_noise(image, mean=0, sigma=0):
    """
    添加高斯噪声
    :param image: PIL Image对象
    :param mean: 噪声均值
    :param sigma: 噪声标准差
    :return: 带噪声的PIL Image
    """
    img_array = np.array(image)
    noise = np.random.normal(mean, sigma, img_array.shape)
    noisy_img = img_array + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

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
    
    with open(answers_file, "w") as ans_file:
        buffer = []  # 初始化缓冲区
        buffer_size = 1000  # 设置缓冲区大小    
        # 逐样本处理（不再使用批处理）
        for line in tqdm(questions, total=len(questions)):
            image_file = line["image"]
            caption = line["text"]
            num_seq = count_valid_sentences(caption)
            
            for s in range(num_seq):
                seq = get_no_k_sentences(caption, s)   
                objects0 = extract_nouns(seq)
                # print(objects)
                obj_blacklist = ["image","addition","atmosphere","others","scene","use","side","environment","work","collaboration","safety","time","center","right","size","something","closer","setting","background","moment","visible","area","place","towards","object","bottom", "corner","mode","experience","activity"]
                objects = []
                for x in objects0:
                    if x not in obj_blacklist and x not in objects:
                        objects.append(x)
                
                # 保存结果
                ans_id = shortuuid.uuid()
                buffer.append(json.dumps({
                        "question_id": line["question_id"],
                        "prompt": line["prompt"],
                        "image": line["image"],
                        "sentence": seq,
                        # "text": line["text"],
                        "objects" : objects,
                        # "answer_id": line["answer_id"],
                        "model_id": line["model_id"]
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
    parser.add_argument("--answers-file", type=str, default="LLaVA_original_0616/exp/0624coco/coco_gen_llava157_n25.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--n_sig", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--startid", type=int, default=0, help="startid")
    parser.add_argument("--endid", type=int, default=0, help="endid")
    parser.add_argument("--part", type=int, default=-1, help="endid")

    args = parser.parse_args()

    eval_model(args)

