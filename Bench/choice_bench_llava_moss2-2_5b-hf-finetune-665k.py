### This file is used to generate the json file for the benchmarking of the model
import sys
import os
import codecs
import argparse

parent_path = os.path.dirname(sys.path[0])
if parent_path not in sys.path:
    sys.path.append(parent_path)


from bench_function import export_distribute_json
import os
import json
import time


if __name__ == "__main__":

    # Load the MCQ_prompt.json file
    with open("MCQ_prompt.json", "r") as f:
        data = json.load(f)['examples']
    f.close()

    model_name = "llava_moss2-2_5b-hf-finetune-665k"
    model_path = "/remote-home/share/models/llava_moss2-2_5b-hf-finetune-665k"

    multi_images = True # whether to support multi images input, True means support, False means not support
        
    for i in range(len(data)):
        directory = "../Data"
        
        keyword = data[i]['keyword']
        question_type = data[i]['type']
        zero_shot_prompt_text = data[i]['prefix_prompt']
        print(model_name)
        print(keyword)
        print(question_type)

        export_distribute_json(
            None,
            model_name, 
            model_path,
            directory, 
            keyword, 
            zero_shot_prompt_text, 
            question_type, 
            multi_images=multi_images
        )

    
