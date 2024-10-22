# Copyright (c) 2023 Alibaba PAI Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import copy
import io
import json
from typing import Dict, Optional, Sequence, List
import numpy as np
import torch
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import transformers
from transformers import Idefics2ImageProcessor, SiglipImageProcessor, CLIPImageProcessor
try:
    from megatron import get_args
except:
    from megatron.training import get_args

from megatron_patch.tokenizer import get_tokenizer
from megatron_patch.data.llava import conversation as conversation_lib
from megatron_patch.data.llava.mm_utils import tokenizer_image_token
from megatron_patch.data.llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from megatron_patch.model.llava.clip_encoder import CLIPVisionTower
from megatron_patch.data.idefics2.idefics2_image_processor import Idefics2ImageProcessorPad

import wids
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial
from packaging import version
import tokenizers
from collections import Counter
import pickle
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')

def bytes_to_img(byte_data):
    #img_stream = io.BytesIO(byte_data)
    img = Image.open(byte_data)
    return img

def btyes_to_dict(byte_data):
    return eval(byte_data.read().decode('utf-8'))

def make_sample(sources, tokenizer, image_processor, data_args):
    #print (sources)
    original_sources = sources
    has_image=True
    if sources['has_image']: #any([name in sources["__key__"] for name in ['coco', 'vqa', 'gqa', 'vg']]): #'/' in sources["__key__"]:
        processor = image_processor
        image = bytes_to_img(sources['.input_image']).convert('RGB')
        images = [image]
        # print ('image size', image.size)
        all_images = []
        all_pixal_masks = []
        for image in images:
            pixal_mask = None
            if data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                try:
                    image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                except:
                    print (sources["__key__"])
                    print (image.size)
                    print ('mode', image.mode, 'mean', processor.image_mean)
                    exit()
                # image = processor.preprocess(image, do_image_splitting=False, size={'height': 336, 'width': 336},return_tensors='pt')['pixel_values'][0]
                # image = image.squeeze(0)
                # image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            #else:
            if data_args.vision_tower.endswith('clip-vit-large-patch14-336'):
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            elif data_args.vision_tower.endswith('idefics2-8b'):
                image_data = processor.preprocess(image, do_image_splitting=False, return_tensors='pt')
                image = image_data['pixel_values'][0]
                image = image.squeeze(0)
                pixal_mask = image_data['pixel_attention_mask'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([btyes_to_dict(sources[".conversations"])]))
            all_images.append(image)
            all_pixal_masks.append(pixal_mask)
    else:
        has_image=False
        sources = copy.deepcopy([btyes_to_dict(sources[".conversations"])])
    if data_args.vision_tower.endswith('clip-vit-large-patch14-336'):
        per_image_token = 576
    elif data_args.vision_tower.endswith('idefics2-8b'):
        per_image_token = 64
    else:
        per_image_token = 729
    data_dict = preprocess_llama3(
        sources,
        tokenizer, num_image=len(all_images) if has_image else 0, per_image_token=per_image_token)
    # if original_sources["__key__"] == 'coco_8750':
    #     print ('IN', sources)
    #     print ('IN', data_dict['input_ids'][0])

    if data_args.max_padding_length <= len(data_dict["input_ids"][0]):
        print ('The example is too long, skipping')
        return None
        # we do this because the data dict only has one example
        data_dict["input_ids"] = data_dict["input_ids"][:, :data_args.max_padding_length]
        data_dict["labels"] = data_dict["labels"][:, :data_args.max_padding_length]
        data_dict["weights"] = data_dict["weights"][:, :data_args.max_padding_length]
        data_dict["preferences"] = data_dict["preferences"][:, :data_args.max_padding_length]
    # else:
    #     padding_len = data_args.max_padding_length-len(data_dict["input_ids"][0])
    #     data_dict["input_ids"] = torch.cat([data_dict["input_ids"], torch.full((1, padding_len), IGNORE_INDEX, dtype=torch.int)], dim=1)
    #     data_dict["labels"] = torch.cat([data_dict["labels"], torch.full((1, padding_len), IGNORE_INDEX, dtype=torch.int)], dim=1)
    data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0], weights=data_dict["weights"][0], preferences=data_dict["preferences"][0])

    # image exist in the data
    if has_image:  # '/' in sources["__key__"]:
        #data_dict['image'] = image.to(dtype=torch.half if data_args.fp16 else torch.bfloat16)
        if data_args.fp16:
            dtype = torch.half
        elif data_args.bf16:
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
        data_dict['image'] = torch.stack(all_images, dim=0).to(dtype=dtype)
        if all_pixal_masks[0] is not None:
            data_dict['pixel_attention_mask'] = torch.cat(all_pixal_masks, dim=0).to(dtype=dtype)
    # else:
    #     # image does not exist in the data, but the model is multimodal
    #     crop_size = image_processor.crop_size
    #     data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width']).to(dtype=torch.half if data_args.fp16 else torch.bfloat16)
    return data_dict

def packing_examples(examples, tokenizer, max_length):
    packed_results = []
    tmp = []
    curr = 0
    packed_has_image = []
    tmp_has_image = []
    for i, exp in tqdm(enumerate(examples)):
        conv = [exp['conversations']]
        if 'image' in exp:
            conv = preprocess_multimodal(conv)
        tokenized = preprocess_llama3(conv, tokenizer)
        this_len = len(tokenized['input_ids'][0])
        # if exp['uid'] in ['coco_22576', 'vgrkXIk_0', 'coco_22577', 'coco_22578', 'coco_22579', 'coco_22580']:
        #     print (exp['uid'], this_len, curr, tmp)
            #print (tokenized['input_ids'][0])
        #print (exp['uid'], this_len)
        if curr + this_len > max_length:
            packed_results.append(tmp)
            packed_has_image.append(tmp_has_image)
            #print (tmp)
            tmp = []
            tmp_has_image = []
            curr = 0
        tmp.append(exp['uid'])
        if 'image' in exp:
            tmp_has_image.append(True)
        else:
            tmp_has_image.append(False)
        curr += this_len
    if len(tmp) > 0:
        packed_results.append(tmp)
        packed_has_image.append(tmp_has_image)
    return packed_results, packed_has_image

class PackedShardListDataset(wids.ShardListDataset):
    def __init__(
        self,
        shards,
        text_data_path,
        tokenizer,
        max_length,
        cache_size=int(1e12),
        cache_dir=None,
        dataset_name=None,
        localname=None,
        transformations="PIL",
        keep=False,
        base=None,
        options=None,
    ):
        super(PackedShardListDataset, self).__init__(shards, cache_size=cache_size, cache_dir=cache_dir, dataset_name=dataset_name, localname=localname, transformations=transformations, keep=keep, base=base, options=options)
        self.tokenizer = tokenizer
        self.max_length = max_length
        data = json.load(open(text_data_path, 'r'))
        self.id2index = {}
        image_count = Counter()
        self.has_image = []
        for i in range(len(data)):
            if 'image' not in data[i] and 'images' not in data[i]:
                self.id2index[data[i]['id']] = i
                data[i]['uid'] = data[i]['id']
                self.has_image.append(False)
            else:
                image_dir = data[i]['image'].split('/')[0]
                image_count[image_dir] += 1
                self.id2index[f"{image_dir}_{image_count[image_dir]}"] = i
                data[i]['uid'] =f"{image_dir}_{image_count[image_dir]}"
                self.has_image.append(True)
        print ("size of id2index", len(self.id2index))
        # num_processes = 32  
        # chunk_size = len(data) // num_processes + 1 
        # partial_packing = partial(packing_examples, tokenizer=tokenizer, max_length=max_length)
        # data_chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        # with Pool(num_processes) as pool:
        #     results = list(pool.imap(partial_packing, data_chunks))

        # self.packed_samples = [item for sublist in results for item in sublist[0]]
        # self.has_image = [item for sublist in results for item in sublist[1]]
        # self.total_length = len(self.packed_samples)
        # print ('Total packed samples', self.total_length)

    # def __len__(self):
    #     """Return the total number of samples in the dataset."""
    #     return 8

    def __getitem__(self, index):
        """Return the sample corresponding to the given index."""
        shard, inner_idx, desc = self.get_shard(index)
        try:
            sample = shard[inner_idx]
        except:
            print (index, inner_idx, desc, 'length of shard', len(shard))

        self.check_cache_misses()

        sample["__dataset__"] = desc.get("dataset")
        sample["__index__"] = index
        sample["__shard__"] = desc["url"]
        sample["__shardindex__"] = inner_idx
        sample["has_image"] = self.has_image[index]

        for transform in self.transformations:
            sample = transform(sample)
        return sample

    # def __getitem__(self, index):
    #     """Return the sample corresponding to the given index."""
    #     covered_examples = self.packed_samples[index]
    #     covered_has_image = self.has_image[index]
    #     covered_sample = []
    #     for _, eid in enumerate(covered_examples):
    #         i = self.id2index[eid]
    #         shard, inner_idx, desc = self.get_shard(i)
    #         sample = shard[inner_idx]

    #         # Check if we're missing the cache too often.
    #         self.check_cache_misses()

    #         sample["__dataset__"] = desc.get("dataset")
    #         sample["__index__"] = i
    #         sample["__shard__"] = desc["url"]
    #         sample["__shardindex__"] = inner_idx
    #         sample["has_image"] = covered_has_image[_]
    #         # Apply transformations
    #         for transform in self.transformations:
    #             sample = transform(sample)
    #         covered_sample.append(sample)
    #     input_ids = torch.cat([sample['input_ids'] for sample in covered_sample], dim=0)
    #     labels = torch.cat([sample['labels'] for sample in covered_sample], dim=0)
    #     weights = torch.cat([sample['weights'] for sample in covered_sample], dim=0)
    #     preferences = torch.cat([sample['preferences'] for sample in covered_sample], dim=0)
    #     #print ('index', index, covered_examples)
    #     if len(input_ids) < self.max_length:
    #         padding_len = self.max_length-len(input_ids)
    #         input_ids = torch.cat([input_ids, torch.full((padding_len,), IGNORE_INDEX, dtype=torch.int)], dim=0)
    #         labels = torch.cat([labels, torch.full((padding_len,), IGNORE_INDEX, dtype=torch.int)], dim=0)
    #         weights = torch.cat([weights, torch.full((padding_len,), 0.0, dtype=torch.float)], dim=0)
    #         preferences = torch.cat([preferences, torch.full((padding_len,), 0, dtype=torch.long)], dim=0)
    #     elif len(input_ids) > self.max_length:
    #         input_ids = input_ids[:self.max_length]
    #         labels = labels[:self.max_length]
    #         weights = weights[:self.max_length]
    #         preferences = preferences[:self.max_length]
    #         print ('Too LONG covered examples', covered_examples)
    #         print ([len(sample['input_ids']) for sample in covered_sample])
    #         exit()
    #     covered_image = [sample['image'] for sample in covered_sample if 'image' in sample]
    #     if len(covered_image) > 0:
    #         image = torch.stack(covered_image, dim=0)
    #     else:
    #         image = None
    #     num_positions = input_ids.eq(IMAGE_TOKEN_INDEX).sum().item()
    #     num_images = len(image) if image is not None else 0
    #     if num_positions != num_images * 576:
    #         print ('MISMATCH', num_positions, num_images, covered_examples)
    #     packed_sample = {
    #         'input_ids': input_ids,
    #         'labels': labels,
    #         'image': image,
    #         'weights': weights,
    #         'preferences': preferences
    #     }
    #     return packed_sample

def get_sft_dataset(data_path):
    tokenizer = get_tokenizer()
    data_args = get_args()
    # vision_tower = CLIPVisionTower(data_args.vision_tower, data_args.cvcuda_image_processing)
    # vision_tower.to(torch.half if data_args.fp16 else torch.bfloat16)
    # image_processor = vision_tower.image_processor
    if data_args.vision_tower.endswith('clip-vit-large-patch14-336'):
        image_processor = CLIPImageProcessor.from_pretrained(data_args.vision_tower)
    elif data_args.vision_tower.endswith('idefics2-8b'):
        #image_processor = Idefics2ImageProcessor.from_pretrained(data_args.vision_tower)
        image_processor = Idefics2ImageProcessorPad.from_pretrained(data_args.vision_tower)
        image_processor.length = image_processor.size['longest_edge']
        print ('Loading Idefics2ImageProcessor from', data_args.vision_tower)
    else:
        image_processor = SiglipImageProcessor.from_pretrained(data_args.vision_tower)
        print ('Loading SigLipImageProcessor from', data_args.vision_tower)
    partial_make_sample = partial(make_sample, tokenizer=tokenizer, image_processor=image_processor, data_args=data_args)
    shardlist = json.load(open(f"{data_path}/shardlist.json"))
    def convert_localname(url):
        return f"{data_path}/{url}"
    localname = {k['url']: f"{data_path}/{k['url']}" for k in shardlist}
    #train_dataset = wids.ShardListDataset(shardlist, localname=convert_localname, keep=True)
    #train_dataset.add_transform(partial_make_sample)
    train_dataset = PackedShardListDataset(shardlist, data_args.image_folder, tokenizer, data_args.max_padding_length, localname=convert_localname, keep=True, transformations=partial_make_sample)
    return train_dataset

def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def _mask_targets(target, tokenized_lens, speakers):
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len

def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation

def preprocess_multimodal(sources: Sequence[str]) -> Dict:
    args = get_args()
    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                # if "mmtag" in conversation_lib.default_conversation.version:
                #     sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources

def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + '<|eot_id|>' #conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX
    input_ids = torch.stack(input_ids, dim=0)
    targets = torch.stack(targets, dim=0)
    return dict(input_ids=input_ids, labels=targets)

def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    # print ('conv', conversations)
    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        # print ('rounds', rounds)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            # KM: if tokenizer.legacy is True, then this code is never used
            # if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
            #     round_len -= 1
            #     instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                #print ('Target', target, f"mismatch {cur_len} vs. {total_len}.")
                # for i, rou in enumerate(rounds):
                #     print ('round', i, tokenizer_image_token(rou, tokenizer),f"mismatch {cur_len} vs. {total_len}.")
                target[:] = IGNORE_INDEX
                #exit()
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
                
        #exit()
    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_llama3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    num_image: int = 1,
    per_image_token: int = 576,
) -> Dict:
    
    mapping = {'human': 'user', 'gpt': 'assistant'}
    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if 'from' in sources[0] and source[0]["from"] != 'human':
            # Skip the first one if it is not from human
            source = source[1:]
        image_token_count = sum([message['content'].count(DEFAULT_IMAGE_TOKEN) if 'content' in message else message['value'].count(DEFAULT_IMAGE_TOKEN) for message in source])
        if num_image > image_token_count:
            source[0]['content'] = DEFAULT_IMAGE_TOKEN * (num_image - image_token_count) + source[0]['content']
        messages = []
        for j, sentence in enumerate(source):
            role = mapping[sentence["from"]] if "from" in sentence else sentence["role"]
            if "value" in sentence:
                messages.append((role, sentence["value"].replace(DEFAULT_IMAGE_TOKEN, '<|reserved_special_token_250|>'*per_image_token)))
            else:
                messages.append((role, sentence['content'].replace(DEFAULT_IMAGE_TOKEN, '<|reserved_special_token_250|>'*per_image_token)))
        conversations.append(messages)
    # print ('conv', conversations)
    # Tokenize conversations

    input_ids = []
    targets = []
    weights = []
    preferences = []
    for messages in conversations:
        this_ids = [tokenizer.bos_token_id]
        this_labels = [tokenizer.bos_token_id]
        this_weights = [0.0]
        this_preferences = [0]
        for role, message in messages:
            head = tokenizer(f"<|start_header_id|>{role}<|end_header_id|>\n\n", add_special_tokens=False).input_ids
            tail = tokenizer(f"{message}<|eot_id|>", add_special_tokens=False).input_ids
            this_ids.extend(head+tail)
            this_labels.extend(head+tail)
            # if role in ['system', 'user']:
            #     this_labels += [IGNORE_INDEX]*(len(head) + len(tail))
            # else:
            #     this_labels += [IGNORE_INDEX]*len(head) + tail
            if role in ['玩家', 'system', "策略", "领域知识", "长期记忆", "user"]:
                this_weights.extend([0.0] * len(head+tail))
                this_preferences.extend([0] * len(head+tail))
            else:
                this_weights.extend([0.0] * len(head))
                this_preferences.extend([0] * len(head))
                # if 'loss' in turn:
                #     tmp_weights.extend([float(turn['loss'])]*left_token_len)
                # else:
                this_weights.extend([1.0] * len(tail))
                # if 'label' in turn:
                #     tmp_preferences.extend([int(turn['label'])] * left_token_len)
                # else:
                this_preferences.extend([2] * len(tail))
        this_ids = [x if x != 128255 else IMAGE_TOKEN_INDEX for x in this_ids]
        # this_ids.append(tokenizer.eos_token_id)
        # this_labels.append(tokenizer.eos_token_id)
        # this_weights.append(0.0)
        # this_preferences.append(0)
        input_ids.append(torch.tensor(this_ids, dtype=torch.long))
        targets.append(torch.tensor(this_labels, dtype=torch.long))
        weights.append(torch.tensor(this_weights, dtype=torch.float))
        preferences.append(torch.tensor(this_preferences, dtype=torch.long))
    input_ids = torch.stack(input_ids, dim=0)
    targets = torch.stack(targets, dim=0)
    weights = torch.stack(weights, dim=0)
    preferences = torch.stack(preferences, dim=0)
    return dict(
        input_ids=input_ids,
        labels=targets,
        weights=weights,
        preferences=preferences,
    )

def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]
    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)

class LazySupervisedDataset(torch.utils.data.Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str):
        super(LazySupervisedDataset, self).__init__()
        self.args = get_args()
        if data_path[0].endswith('jsonl'):
            self.list_data_dict = [json.loads(line) for line in open(data_path[0], "r", encoding='utf-8')]
        else:
            self.list_data_dict = json.load(open(data_path[0], "r"))
        print (f"Loaded {len(self.list_data_dict)} samples from {data_path[0]}")
        self.tokenizer = get_tokenizer()

        if self.args.vision_tower.endswith('clip-vit-large-patch14-336'):
            vision_tower = CLIPVisionTower(self.args.vision_tower, self.args.cvcuda_image_processing)
            vision_tower.to(torch.half if self.args.fp16 else torch.bfloat16)
            self.image_processor = vision_tower.image_processor
        elif self.args.vision_tower.endswith('idefics2-8b'):
            self.image_processor = Idefics2ImageProcessor.from_pretrained(self.args.vision_tower)
            print ('Loading Idefics2ImageProcessor from', self.args.vision_tower)
        else:
            self.image_processor = SiglipImageProcessor.from_pretrained(self.args.vision_tower)
            print ('Loading SigLipImageProcessor from', self.args.vision_tower)
            #raise ValueError(f'Unknown vision tower: {self.args.vision_tower}')

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'images' in sources[0] or 'image' in sources[0]:
            image_files = [self.list_data_dict[i]['image']] if 'image' in self.list_data_dict[i] else self.list_data_dict[i]['images']
            image_folder = self.args.image_folder
            processor = self.image_processor
            all_images = []
            for image_file in image_files:
                image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
                if self.args.image_aspect_ratio == 'pad':
                    def expand2square(pil_img, background_color):
                        width, height = pil_img.size
                        if width == height:
                            return pil_img
                        elif width > height:
                            result = Image.new(pil_img.mode, (width, width), background_color)
                            result.paste(pil_img, (0, (width - height) // 2))
                            return result
                        else:
                            result = Image.new(pil_img.mode, (height, height), background_color)
                            result.paste(pil_img, ((height - width) // 2, 0))
                            return result
                    image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                else:
                    #image = processor.preprocess(image, do_image_splitting=False, size={'height': 336, 'width': 336}, return_tensors='pt')['pixel_values'][0]
                    image = processor.preprocess(image, size={'height': 336, 'width': 336}, return_tensors='pt')['pixel_values'][0]
                    image = image.squeeze(0)
                all_images.append(image)
            # # This put all inputs at the beginning
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] if "conversations" in e else e["messages"] for e in sources]))
        else:
            all_images = None
            sources = copy.deepcopy([e["conversations"] if "conversations" in e else e["messages"] for e in sources])
        if self.args.vision_tower.endswith('clip-vit-large-patch14-336'):
            per_image_token = 576
        elif self.args.vision_tower.endswith('idefics2-8b'):
            per_image_token = 64
        else:
            per_image_token = 729
        data_dict = preprocess_llama3(
            sources,
            self.tokenizer, per_image_token=per_image_token)
        if self.args.max_padding_length <= len(data_dict["input_ids"][0]):
            print ('The example is too long, skipping')
            return None
            data_dict["input_ids"] = data_dict["input_ids"][:, :self.args.max_padding_length]
            data_dict["labels"] = data_dict["labels"][:, :self.args.max_padding_length]
            data_dict["weights"] = data_dict["weights"][:, :self.args.max_padding_length]
            data_dict["preferences"] = data_dict["preferences"][:, :self.args.max_padding_length]
        # else:
        #     padding_len = self.args.max_padding_length-len(data_dict["input_ids"][0])
        #     data_dict["input_ids"] = torch.cat([data_dict["input_ids"], torch.full((1, padding_len,), IGNORE_INDEX, dtype=torch.int)], dim=1)
        #     data_dict["labels"] = torch.cat([data_dict["labels"], torch.full((1, padding_len,), IGNORE_INDEX, dtype=torch.int)], dim=1)

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0], weights=data_dict["weights"][0], preferences=data_dict["preferences"][0])

        # image exist in the data
        if all_images is not None:
            data_dict['image'] = torch.stack(all_images, dim=0).to(dtype=torch.half if self.args.fp16 else torch.bfloat16)
        # elif self.args.is_multimodal:
        #     # image does not exist in the data, but the model is multimodal
        #     crop_size = self.image_processor.crop_size
        #     data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict
