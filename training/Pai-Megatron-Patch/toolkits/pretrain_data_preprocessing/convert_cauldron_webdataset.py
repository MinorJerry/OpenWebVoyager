import os
import subprocess
from multiprocessing import Pool
import glob
from tqdm import tqdm
import json 
from PIL import Image
from collections import Counter 
import webdataset as wds
import random 
import zipfile
import pyarrow.parquet as pq
import pandas as pd
import io
import pickle
from datasets import load_dataset

def convert_cauldron_to_chat(conv, num_image=1):
    result = [{'role': "user", "content": "<image> " *  num_image+ conv['user']},
        {'role': "assistant", "content": conv['assistant']}]
    return result

# def convert_cauldron_webdataset():

all_data = []

sink = wds.ShardWriter(f"/data7/tianqfang/the_cauldron_webdataset_v1.1/cauldron-%06d.tar", maxcount=25000, maxsize=3e10)
dataset_counter = Counter()
dataset_entry_counter = Counter()
sub_dirs = glob.glob('/data7/tianqfang/the_cauldron/*')
cnt = 0

## process okvqa and clevr_math

dataset = load_dataset("/data7/tianqfang/OK-VQA_train", split='train')

for i, exp in tqdm(enumerate(dataset)):

    output = io.BytesIO()
    exp['image'].save(output, format='JPEG')
    output.seek(0)

    Image.open(output)
    
    img_list = [output]
    img_list = pickle.dumps(img_list)

    result = [{'role': "user", "content": "<image>" + exp['question']},
        {'role': "assistant", "content": exp['answers'][0]}]

    final_uid = f"okvqa_{i}"
    item = {
            "__key__": final_uid,
            "input_image": img_list,
            "conversations": str(result),
        }
    all_data.append(item)
    cnt += 1

# clevr_math

dataset = json.load(open("/data7/tianqfang/clevr_math/data/clevr-math/clevr-math-train.json"))['questions'] + \
    json.load(open("/data7/tianqfang/clevr_math/data/multi-hop/clevr-math-train.json"))['questions']

random.shuffle(dataset)
dataset = dataset[:100000]

for i, exp in tqdm(enumerate(dataset)):
    result = [{'role': "user", "content": "<image>" + exp['question']},
        {'role': "assistant", "content": str(exp['answer'])}]
    img_list = pickle.dumps([io.BytesIO(open("/data7/tianqfang/clevr_math/data/CLEVR_v1.0/images/train/" + exp['image_filename'], "rb").read())])
    final_uid = f"clevr_math_{i}"
    item = {
            "__key__": final_uid,
            "input_image": img_list,
            "conversations": str(result),
        }
    all_data.append(item)
    cnt += 1



for dir in tqdm(sub_dirs):
    if not os.path.isdir(dir):
        continue
        
    # temporary
    # if not "ai2d" in dir:
    #     continue
    print (dir)
    dir_name = dir.split('/')[-1]
    if dir_name in ['okvqa', 'clevr_math']:
        continue

    parquet_files = glob.glob(f'{dir}/train*.parquet')
    dir_data = []
    for parquet_file in parquet_files:
        table = pq.read_table(parquet_file)
        df = table.to_pandas()
        list_of_dicts = df.to_dict('records')
        dir_data.extend(list_of_dicts)

    for n_data, exp in enumerate(dir_data):
        if dir_name in ["clevr", "clevr_math"]:
            if n_data >= 40000:
                break
        elif dir_name in ['figureqa', "tallyqa"]:
            if n_data >= 50000:
                break
        elif dir_name in ['dvqa', 'localized_narratives']:
            if n_data >= 100000:
                break

        #uid = f"{dir_name}_{exp['id']}"
        img_list = [io.BytesIO(exp['images'][i]['bytes']) for i in range(len(exp['images']))]
        bad_flag = False
        # for img in img_list:
        #     try:
        #         Image.open(img)
        #     except:
        #         bad_flag = True
        # if bad_flag:
        #     print("bad example in ", dir_name)
        #     continue
        image_num=len(img_list)
        img_list = pickle.dumps(img_list)
        dataset_entry_counter[dir_name] += 1

        for i, texts in enumerate(exp['texts']):
            if dir_name in ['dvqa', 'clevr', 'clevr_math', 'figureqa', 'mapqa', 'ocrvqa', 'vqav2']:
                if i >= 3:
                    break
            elif dir_name == 'plotqa':
                if i >=5:
                    break
            dataset_counter[dir_name] += 1
            final_uid = f"{dir_name}_{dataset_counter[dir_name]}"
            cnt += 1
            all_data.append({
                    "__key__": final_uid,
                    "input_image": img_list,
                    "conversations": str(list(convert_cauldron_to_chat(texts, image_num))),
                })
    
print("dataset_counter", dataset_counter)
print("dataset_entry_counter", dataset_entry_counter)
print("total number of MM data", cnt)

base_dir = '/data7/tianqfang/'
text_datasets = ['lima', 'atlas-math-sets', 'databricks-dolly-15k', 'goat', 'MathInstruct', 'MetaMathQA', 'OpenHermes-2.5', 'camelaimath']
# text_datasets = ['lima']
cnt = 0
for dir_name in tqdm(text_datasets):
    path = os.path.join(base_dir, dir_name)
    if dir_name == 'camelaimath':
        files = glob.glob(path + "/*.json")
        for i, f in tqdm(enumerate(files)):
            exp = json.load(open(f))
            result = [{'role': "user", "content": exp['message_1']},
            {'role': "assistant", "content": exp['message_2']}]

            final_uid = f"{dir_name}_{i}"
            item = {
                    "__key__": final_uid,
                    "conversations": str(result).replace("<image>", "< image >"),
                }
            all_data.append(item)
            cnt += 1

        continue
    
    dataset = load_dataset(path)

    if dir_name == 'lima':
        for i, exp in enumerate(dataset['train']):
            result = [{'role': "user", "content": exp['conversations'][0]},
            {'role': "assistant", "content": exp['conversations'][1]}]

            final_uid = f"{dir_name}_{i}"
            item = {
                    "__key__": final_uid,
                    "conversations": str(result).replace("<image>", "< image >"),
                }
            all_data.append(item)
            cnt += 1
    elif dir_name in ['atlas-math-sets', 'goat', 'MathInstruct']:
        for i, exp in enumerate(dataset['train']):
            if i >= 50000:
                break
            # if dir_name == 'goat':
            #     if i >= 50000:
            #         break
            # elif dir_name == 'MathInstruct':
            #     if i >= 50000:
            #         break
            # elif dir_name == 'atlas-math-sets':
            #     if i >= 50000:
            #         break

            result = [{'role': "user", "content": exp['instruction']},
            {'role': "assistant", "content": exp['output']}]

            final_uid = f"{dir_name}_{i}"
            item = {
                    "__key__": final_uid,
                    "conversations": str(result).replace("<image>", "< image >"),
                }
            all_data.append(item)
            cnt += 1
    elif dir_name == 'databricks-dolly-15k':

        for i, exp in enumerate(dataset['train']):
            if i >= 10000:
                break
            result = [{'role': "user", "content": exp['instruction'] + "\n" + exp['context']},
            {'role': "assistant", "content": exp['response']}]

            final_uid = f"{dir_name}_{i}"
            item = {
                    "__key__": final_uid,
                    "conversations": str(result).replace("<image>", "< image >"),
                }
            cnt += 1
            all_data.append(item)
    elif dir_name == 'MetaMathQA':
        for i, exp in enumerate(dataset['train']):
            if i >= 50000:
                break
            result = [{'role': "user", "content": exp['query']},
            {'role': "assistant", "content": exp['response']}]

            final_uid = f"{dir_name}_{i}"
            item = {
                    "__key__": final_uid,
                    "conversations": str(result).replace("<image>", "< image >"),
                }
            cnt += 1
            all_data.append(item)
    elif dir_name == 'OpenHermes-2.5':
        for i, exp in enumerate(dataset['train']):
            if i >= 200000:
                break
            result = [{'role': "user", "content": exp['conversations'][0]['value']},
            {'role': "assistant", "content": exp['conversations'][1]['value']}]

            final_uid = f"{dir_name.replace('.', '')}_{i}"
            item = {
                    "__key__": final_uid,
                    "conversations": str(result).replace("<image>", "< image >"),
                }
            cnt += 1
            all_data.append(item)

print("total number of text data", cnt)

print("saving shuffled data to disk")
random.shuffle(all_data)
for item in tqdm(all_data):
    sink.write(item)

shardlist = []
shard = 0
for start in range(0, cnt, 25000):
    shardlist.append({'url': f"cauldron-{shard:06d}.tar", 'nsamples': min(25000, cnt - start)})
    shard += 1
with open('/data7/tianqfang/the_cauldron_webdataset_v1.1/shardlist.json', 'w') as fout:
    json.dump(shardlist, fout, indent=4)

# 
# convert_cauldron_webdataset()



######## SANITY CHECK

import copy
from itertools import chain
def rename_dict(d):
    new_d = {}
    role_dict = {'human':'user', 'gpt':'assistant'}
    new_d['role'] = role_dict[d['from']]
    new_d['content'] = d['value']
    return new_d

from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained("/path/to",
                                          do_image_splitting=False)

for sources in tqdm(all_data):
    # has_image=True
    # if sources['has_image']: 

    if 'input_image' in sources:
        images = pickle.loads(sources['input_image'])
        images = [Image.open(image).convert('RGB') for image in images]
        # images = [1 for image in images]
    else:
        images = [None]
    sources = copy.deepcopy([eval(sources["conversations"])])

    # FIXME:
    # the format of the previous version
    if 'from' in sources[0][0]:
        sources = [[rename_dict(d) for d in sources[0]]]

    # if data_args.max_padding_length <= len(data_dict["input_ids"][0]):
    #     print ('The example is too long, skipping')
    #     return None

    # pretraining
    assert len(sources[0]) % 2 == 0
    assert all([sources[0][2*i]['role'] == 'user' for i in range(len(sources[0])//2)])
    assert all([sources[0][2*i + 1]['role'] == 'assistant' for i in range(len(sources[0])//2)])
    
    # example = {"image": image, 'question':question, 'answer':answer}

    # number of images must be equal to the number of <image> tokens
    # print(len(images), question.count("<image>"))

    all_queries = [sources[0][2*i]['content'] for i in range(len(sources[0])//2)]
    all_answers = [sources[0][2*i + 1]['content'] for i in range(len(sources[0])//2)]

    if all([img is None for img in images]):
        assert 0 == sum([question.count("<image>") for question in all_queries]), sources
    else:
        assert sum([img is not None for img in images]) == sum([question.count("<image>") for question in all_queries])
    
    texts = []

    messages = list(chain(*[
                [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": question}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": answer}
                        ]
                    }
                ]
                for question, answer in zip(all_queries, all_answers)
            ]
        ))
    
    text = processor.apply_chat_template(messages, add_generation_prompt=False)
    

    texts.append(text.strip())

    # FIXME:
    # processor doesn't support passing do_image_splitting as arguments.
    # manually set it for now.

    # change it to self.do_image_splitting


    if images[0]:
        batch = processor(text=texts, images=images, return_tensors="pt", padding='max_length', max_length=1024, truncation=True)
    else:
        # no image at all
        batch = processor(text=texts, return_tensors="pt", padding='max_length', max_length=1024, truncation=True)
    