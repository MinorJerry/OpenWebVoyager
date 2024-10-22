# -*- coding: utf-8 -*-

import os
import re
import json
import torch
import pickle
import argparse
from fastapi import FastAPI
from pydantic import BaseModel, conbytes
import base64
from PIL import Image
from transformers import AutoProcessor, Idefics2ForConditionalGeneration
import uvicorn
from fastapi.middleware.cors import CORSMiddleware


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--model_name_or_path", type=str,
                    default="model")
parser.add_argument("--port", type=int, default=8080)
args = parser.parse_args()
# full_model_path = "/path/to/hf_ckp/"
# args.model_name_or_path = full_model_path
print(args)

print("Current loaded model:", args.model_name_or_path.split("/")[-3])

processor = AutoProcessor.from_pretrained(
    args.model_name_or_path,
    do_image_splitting=False
)
processor.image_processor.size['longest_edge'] = 980
processor.image_processor.size['shortest_edge'] = 980

class MyDataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<image>")
        ]

    def __call__(self, example_list):
        texts = []
        images = []
        for example in example_list:
            image_list = example["images"]
            
            messages = []
            conversations = example["conversations"]
            for conv in conversations:
                item = {}
                item["role"] = conv["role"]
                raw_content = conv["content"]
                raw_content_split = re.split(r'(<image>)', raw_content)
                content_list = [{"type": "image"} if seg == "<image>"
                            else {"type": "text", "text": seg} for seg in raw_content_split]
                item["content"] = content_list
                messages.append(item)
        

            text = processor.apply_chat_template(messages, add_generation_prompt=True)
            texts.append(text.strip())
            images.append(image_list)

        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

        return batch

model = Idefics2ForConditionalGeneration.from_pretrained(
    args.model_name_or_path,
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",  # Only available on A100 or H100
).to(args.device)


data_collator = MyDataCollator(processor)

class InputData(BaseModel):
    id: str
    conversations: list
    images: str

class OutputPrediction(BaseModel):
    generated_text: str


app = FastAPI()
@app.post("/predict")
def predict(example: InputData):
    example = example.dict()
    image_list_bin = base64.b64decode(example["images"])
    image_list = pickle.loads(image_list_bin)
    example["images"] = image_list
    # example.image = [str(type(img)) for img in example.image]
    print('finish')
    batch = data_collator([example])
    batch = {k: v.to(args.device) for k, v in batch.items()}
    with torch.no_grad():
        generated_ids = model.generate(**batch, max_new_tokens=256, min_new_tokens=3, 
                                eos_token_id=processor.tokenizer.convert_tokens_to_ids('<end_of_utterance>'), do_sample=True, temperature=1.2)
    generated_text = processor.batch_decode(generated_ids[:, batch["input_ids"].size(1):], skip_special_tokens=True)
    generated_text = generated_text[0]
    
    input_token_count = batch["input_ids"].size(1)
    output_token_count = generated_ids.size(1) - input_token_count
    return {"text": generated_text, "prompt_tokens": input_token_count, "completion_tokens": output_token_count}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)
