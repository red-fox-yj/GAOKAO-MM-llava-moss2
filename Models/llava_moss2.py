import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# parameters
model_path = "/remote-home/share/models/llava_moss2-2_5b-hf-finetune-665k"
dtype = torch.float16

# forward
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=dtype, device_map="auto")
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

## generate
data = [
    "回答这些问题，你的答案应该尽可能简单，以提示“答案是”开始你的答案。\n问：“子曰：学而时习之”的下一句是？\n答：",
    "拥被寒逾峭，雨声遥夜来。寂寞秋在念，芳菲发先衰。采采一寸芹，靡靡不食饱。王孙从何来，芳年亦同老。徒忧侵病颜，积霭迷远抱。犹能尊酒期，共语山中好。请参照这首诗的意境，为以下诗句续写下一句：今人思古人\n答：",
    "Long long ago, there is a princess that lives in a"
    ]
results = []
prompts = []

for d in data:
    input_ids = tokenizer(d, return_tensors="pt")["input_ids"]
    generate = model.generate(input_ids.cuda(), temperature=0.0, num_beams=1, max_new_tokens=100, top_p=1.0, top_k=50, do_sample=False, repetition_penalty=1.0,)
    res = tokenizer.decode(generate[0][input_ids.shape[1]:].tolist(), skip_special_tokens=True)
    results.append(res)
    prompts.append(d)
    print(input_ids)
    print(res)
    print(generate[0].tolist())
    print("====================================================")