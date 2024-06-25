### This file contains the functions for benchmarking the performance of the LLM on multiple-choice questions
import base64
import os
import json
import sys
import time
import re
from random import choice
import requests
from typing import List, Union, Dict
# from joblib import Parallel, delayed
import codecs

import torch
from tqdm import  tqdm
from PIL import Image

# 将父目录添加到 sys.path 中
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from llava.model.builder import load_pretrained_model
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path


def extract_choice_answer(model_output, question_type, answer_lenth=None):
    """
    Extract choice answer from model output.

    Format of model_output that is expected:
    'single_choice': choice answer should be the last Capital Letter of the model_output, e.g.: "...【答案】 A <eoa>"
    'multi_choice': "...【答案】 ABD " or write the choice answers at the end of the model_output, e.g. "... ACD"
    """
    if question_type == 'single_choice':
        model_answer = []
        temp = re.findall(r'[A-D]', model_output[::-1])
        if len(temp) != 0:
            model_answer.append(temp[0])

    elif question_type == 'multi_choice':
        model_answer = []
        answer = ''
        content = re.sub(r'\s+', '', model_output)
        answer_index = content.find('【答案】')
        if answer_index > 0:
            temp = content[answer_index:]
            if len(re.findall(r'[A-D]', temp)) > 0:
                for t in re.findall(r'[A-D]', temp):
                    answer += t
        else:
            temp = content[-10:]
            if len(re.findall(r'[A-D]', temp)) > 0:
                for t in re.findall(r'[A-D]', temp):
                    answer += t
        if len(answer) != 0:
            model_answer.append(answer)
    
    return model_answer

def choice_test(**kwargs):
    """
    Test the LLM on multiple-choice questions
    """
    model_api = kwargs['model_api']
    model_name = kwargs['model_name']
    model_path = kwargs['model_path']
    
    data = kwargs['data']['example']
    keyword = kwargs['keyword']
    prompt = kwargs['prompt']
    question_type = kwargs['question_type']
    multi_images = kwargs['multi_images']
    
    save_dir = f'../Results/{model_name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_file = os.path.join(save_dir, f'{model_name}_{keyword}.json')

    if os.path.exists(save_file):
        with open(save_file, 'r') as f:
            model_answer_dict = json.load(f)['example']
    else:
        model_answer_dict = []


    # 提前加载模型
    if(model_name in ['llava_moss2-2_5b-hf-finetune-150k','llava_moss2-2_5b-hf-finetune-665k']):
            tokenizer, model, image_processor, context_len  = load_pretrained_model(model_path, None, "llava_moss2")
    for i in tqdm(range(len(data))):
        if model_answer_dict != [] and i <= model_answer_dict[-1]['index']:
            continue

        index = data[i]['index']
        question = data[i]['question'].strip() + '\n'
        picture = data[i]['picture']
        year = data[i]['year']
        category = data[i]['category']
        score = data[i]['score']
        standard_answer = data[i]['answer']
        answer_lenth = len(standard_answer)
        analysis = data[i]['analysis']

        if multi_images is False and len(picture) > 1:
            continue

        # llava_moss2不采用api调用，特殊处理
        if(model_name in ['llava_moss2-2_5b-hf-finetune-150k','llava_moss2-2_5b-hf-finetune-665k']):
            # 处理文本
            if model.config.mm_use_im_start_end:
                question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
            else:
                question = DEFAULT_IMAGE_TOKEN + '\n' + question

            conv = conv_templates["llava_llama_2"].copy()
            # 对话模板中加载系统消息
            conv.system = prompt

            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            generate_prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(generate_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
            input_ids = input_ids.to(device='cuda', non_blocking=True)

            # 处理图像
            image = Image.open(picture[0]).convert('RGB')
            image_tensor = process_images([image], image_processor, model.config)[0].unsqueeze(0)
            
            # 推理
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                    image_sizes=image.size,
                    do_sample= False,
                    temperature=0,
                    max_new_tokens=200,
                    use_cache=True)

            model_output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        else:
            model_output = model_api(prompt, question, picture)
        model_answer = extract_choice_answer(model_output, question_type, answer_lenth)

        dict = {
            'index': index, 
            'year': year, 
            'category': category,
            'score': score,
            'question': question, 
            'standard_answer': standard_answer,
            'analysis': analysis,
            'model_answer': model_answer,
            'model_output': model_output
        }
        model_answer_dict.append(dict)

        time.sleep(1)

        with codecs.open(save_file, 'w+', 'utf-8') as f:
            output = {
                'keyword': keyword, 
                'model_name': model_name,
                'prompt': prompt,
                'example' : model_answer_dict
                }
            json.dump(output, f, ensure_ascii=False, indent=4)
            f.close()

def encode_image(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def export_distribute_json(
        model_api:None,
        model_name: str,  
        model_path: None,
        directory: str, 
        keyword: str, 
        zero_shot_prompt_text: str, 
        question_type: str,
        multi_images: bool = True,
    ) -> None:
    """
    Distributes the task of processing examples in a JSON file across multiple processes.

    :param model_name: Name of the model to use
    :param directory: Directory containing the JSON file
    :param keyword: Keyword used to identify the JSON file
    :param zero_shot_prompt_text: Prompt text for zero-shot learning
    :param question_type: Type of questions in the JSON file (e.g. single_choice, five_out_of_seven, etc.)
    :param multi_images: Whether the LLM support multiple images inputs
    
    """
    # Find the JSON file with the specified keyword
    for root, _, files in os.walk(directory):
        for file in files:
            if file == f'{keyword}.json':
                filepath = os.path.join(root, file)
                with codecs.open(filepath, 'r', 'utf-8') as f:
                    data = json.load(f)
        
    
    kwargs = {
        'model_api': model_api,
        'model_name': model_name, 
        'model_path': model_path,
        'data': data, 
        'keyword': keyword, 
        'prompt': zero_shot_prompt_text, 
        'question_type': question_type, 
        'multi_images': multi_images,
    }
    
    if question_type in ["single_choice", "multi_choice"]:
            choice_test(**kwargs)

