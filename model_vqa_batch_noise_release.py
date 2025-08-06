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

def add_gaussian_noise(image, mean=0, sigma=25):
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

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    try:
        questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
        questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    except:
        questions = json.load(open(args.question_file, 'r'))
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    if args.part >= 0:
        start = len(questions)*(args.part)//8
        end = len(questions)*(args.part+1)//8
        questions = questions[start:end]
    # Set batch size
    batch_size = args.batch_size
    
    with open(answers_file, "w") as ans_file:
        # Process questions in batches
        for batch_questions in tqdm(batch(questions, batch_size), total=math.ceil(len(questions)/batch_size)):
            # Prepare batch data
            image_files = []
            prompts = []
            input_ids_list = []
            image_tensors = []
            image_sizes = []
            
            for line in batch_questions[0]:
                image_file = line["image"]
                qs = line["text"]
                cur_prompt = qs
                
                if model.config.mm_use_im_start_end:
                    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

                conv = conv_templates[args.conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
                
                image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
                if args.n_sig>0:
                    image = add_gaussian_noise(image, sigma=args.n_sig)
                image_tensor = process_images([image], image_processor, model.config)[0]
                
                image_files.append(image_file)
                prompts.append(cur_prompt)
                input_ids_list.append(input_ids)
                image_tensors.append(image_tensor)
                image_sizes.append(image.size)
            
            # Stack batch tensors
            # input_ids = torch.cat(input_ids_list, dim=0).cuda()
            # image_tensors = torch.stack(image_tensors, dim=0).half().cuda()
            input_ids = input_ids_list[0]
            image_tensors = image_tensors[0]       
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensors,
                    image_sizes=image_sizes,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=1024,
                    use_cache=True)

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            
            # Write answers for each sample in the batch
            for idx, line, output in zip([q["id"] for q in batch_questions], batch_questions, outputs):
                ans_id = shortuuid.uuid()
                ans_file.write(json.dumps({
                    "question_id": idx,
                    "prompt": line["text"],
                    "image": line["image"],
                    "text": output.strip(),
                    "answer_id": ans_id,
                    "model_id": model_name,
                    "metadata": {}
                }) + "\n")
                ans_file.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="~/model/llava-1.5-7b")#~/model/liuhaotian/llava1.6-34b
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="data/train2014")
    parser.add_argument("--question-file", type=str, default="data/train2014_caption.jsonl")
    parser.add_argument("--answers-file", type=str, default="exp/llava/coco2014_llava157.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=2.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--n_sig", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference")

    parser.add_argument("--part", type=int, default=-1)

    args = parser.parse_args()
    set_seed(28621125)
    eval_model(args)

