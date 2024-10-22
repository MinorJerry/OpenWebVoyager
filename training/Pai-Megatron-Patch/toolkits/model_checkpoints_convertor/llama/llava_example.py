import argparse
import os
import subprocess

import torch

from vllm import LLM
from vllm.sequence import MultiModalData
from transformers import CLIPImageProcessor, LlavaForConditionalGeneration, AutoTokenizer
from PIL import Image
import io
from vllm.config import VisionLanguageConfig
import pyarrow.parquet as pq
import glob
from tqdm import tqdm
from transformers import GenerationConfig
import json
from mmmu_utils import parse_multi_choice_response, parse_open_response, eval_multi_choice, eval_open
# The assets are located at `s3://air-example-data-2/vllm_opensource_llava/`.

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

def run_llava_pixel_values():
    llm = LLM(
        model="/path/to/weboutput_llava/checkpoint/llava-sft-llama3-8B-lr-2e-5-bs-2-seqlen-4096-pr-bf16-tp-8-pp-1-ac-sel-do-true-sp-true-warmup156_HF/",
        image_input_type="PIXEL_VALUES",
        image_token_id=128200,
        image_input_shape="1,3,336,336",
        image_feature_size=576,
    )
    image_processor = CLIPImageProcessor.from_pretrained('/path/to/openai/clip-vit-large-patch14-336')
    prompt = "<|begin_of_text|>" + "<|reserved_special_token_195|>" * 576 + (
        "<|start_header_id|>user<|end_header_id|>\n\nWhat is the content of this image?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")

    # This should be provided by another online or offline component.
    image = Image.open('000000033471.jpg').convert('RGB')
    image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
    images = image_processor.preprocess(image, return_tensors='pt')['pixel_values']
    print ('images shape:', images.shape)

    outputs = llm.generate(prompt,
                           multi_modal_data=MultiModalData(
                               type=MultiModalData.Type.IMAGE, data=images))
    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)


def run_llava_image_features():
    llm = LLM(
        model="llava-hf/llava-1.5-7b-hf",
        image_input_type="image_features",
        image_token_id=32000,
        image_input_shape="1,576,1024",
        image_feature_size=576,
    )

    prompt = "<image>" * 576 + (
        "\nUSER: What is the content of this image?\nASSISTANT:")

    # This should be provided by another online or offline component.
    images = torch.load("images/stop_sign_image_features.pt")

    outputs = llm.generate(prompt,
                           multi_modal_data=MultiModalData(
                               type=MultiModalData.Type.IMAGE, data=images))
    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)

class LocalLlavaForConditionalGeneration(LlavaForConditionalGeneration):
    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids, attention_mask, labels):
        num_images, num_image_patches, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(self.pad_token_id))
        #print ('image features shape:', image_features.shape)
        #print ('input_ids shape:', input_ids.shape)
        # 1. Create a mask to know where special image tokens are
        special_image_token_mask = input_ids == self.config.image_token_index
        num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
        #print ('num_special_image_tokens:', num_special_image_tokens)
        # Compute the maximum embed dimension
        max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)) + sequence_length
        batch_indices, non_image_indices = torch.where(input_ids != self.config.image_token_index)
        # print (max_embed_dim)
        # exit()
        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged image-text sequence.
        # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
        # `torch.cumsum` computes how each image token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_image_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]
        print ('new_token_positions:', new_token_positions)
        print ('text_to_overwrite:', text_to_overwrite)
        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim), self.config.ignore_index, dtype=input_ids.dtype, device=input_ids.device
            )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_image_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_image_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]

        # 5. Fill the embeddings corresponding to the images. Anything that is not `text_positions` needs filling (#29835)
        image_to_overwrite = torch.full(
            (batch_size, max_embed_dim), True, dtype=torch.bool, device=inputs_embeds.device
        )
        image_to_overwrite[batch_indices, text_to_overwrite] = False
        image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(target_device)
        #print ('image_to_overwrite:', image_to_overwrite)
        #exit()
        if image_to_overwrite.sum() != image_features.shape[:-1].numel():
            raise ValueError(
                f"The input provided to the model are wrong. The number of image tokens is {torch.sum(special_image_token_mask)} while"
                f" the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(target_device)
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)
        print ('final position ids:', position_ids)
        print ('attention mask', final_attention_mask)
        # 6. Mask out the embedding at padding positions, as we later use the past_key_value value to determine the non-attended tokens.
        batch_indices, pad_indices = torch.where(input_ids == self.pad_token_id)
        indices_to_mask = new_token_positions[batch_indices, pad_indices]

        final_embedding[batch_indices, indices_to_mask] = 0

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels, position_ids

def run_llava_local_inference(args):
    all_data = []
    all_files = glob.glob("/path/to/pretrain_data/MMMU/*/val*.parquet")
    for i, f in enumerate(all_files):
        print (f)
        table = pq.read_table(f)
        df = table.to_pandas()
        list_of_dicts = df.to_dict('records')
        all_data.extend(list_of_dicts)
    print (len(all_data), 'in total')
    # get this shard of data 
    shard_size = len(all_data) // args.num_shards + 1
    all_data = all_data[args.shard * shard_size: (args.shard + 1) * shard_size]
    print ('shard', args.shard, len(all_data), 'in this shard', 'working on', args.shard * shard_size, (args.shard + 1) * shard_size)
    llava = LlavaForConditionalGeneration.from_pretrained('/path/to/weboutput_llava/checkpoint/llava-sft-llama3-8B-lr-2e-5-bs-2-seqlen-8192-pr-bf16-tp-8-pp-1-ac-full-do-true-sp-true-mantis_HF/')
    llava.eval()
    llava.to(f'cuda:0')
    tokenizer = AutoTokenizer.from_pretrained('/path/to/weboutput_llava/checkpoint/llava-sft-llama3-8B-lr-2e-5-bs-2-seqlen-8192-pr-bf16-tp-8-pp-1-ac-full-do-true-sp-true-mantis_HF/')
    image_processor = CLIPImageProcessor.from_pretrained('/path/to/openai/clip-vit-large-patch14-336')
    loss_fct = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        matches = []
        results = []
        symbols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
        ans_mapping = {s: i for i, s in enumerate(symbols)}
        #print ('ans_mapping', ans_mapping)
        for exp in tqdm(all_data):
            question = exp['question']
            options = eval(exp['options'])
            if exp['question_type'] == 'multiple-choice':
                #print (options)
                #prompt = "<|begin_of_text|>" + f"<|start_header_id|>user<|end_header_id|>\n\n<|reserved_special_token_195|>{question}\n\n" 
                prompt = "<|begin_of_text|>" + f"<|start_header_id|>user<|end_header_id|>\n\n{question}\n\n" 
                for i, option in enumerate(options):
                    prompt += f"({symbols[i]}) {option}\n"
                prompt += "\n\nAnswer with the option's letter from the given choices directly.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            else:
                #prompt = "<|begin_of_text|>" + f"<|start_header_id|>user<|end_header_id|>\n\n<|reserved_special_token_195|>{question}\n\nAnswer the question using a single word or phrase.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                prompt = "<|begin_of_text|>" + f"<|start_header_id|>user<|end_header_id|>\n\n{question}\n\nAnswer the question using a single word or phrase.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            images = []
            for i in range(7):
                if exp[f"image_{i+1}"] is not None and f"<image {i+1}>" in prompt:
                    image = Image.open(io.BytesIO(exp[f"image_{i+1}"]['bytes'])).convert('RGB')
                    images.append(image)
                    prompt = prompt.replace(f"<image {i+1}>", "<|reserved_special_token_195|>")
                    #prompt = prompt.replace(f"<image {i+1}>", "")
            #try:
            images = [expand2square(image, tuple(int(x*255) for x in image_processor.image_mean)) for image in images]
            # if len(images) > 1:
            #     continue 
            #except:
                # print ([image.shape for image in images])
                # continue
                # print ('expansion filled')
                # print (exp['id'])
                # exit()
                # pass
            images = torch.cat([image_processor.preprocess(image, return_tensors='pt')['pixel_values'] for image in images], dim=0)
            images = images.to(llava.device)
            #print ('images shape:', images.shape)
            #print (question)
            #print (prompt)
            # caption_prompt = "<|begin_of_text|>" + "<|reserved_special_token_195|>" + "<|start_header_id|>user<|end_header_id|>\n\nWhat is the content of this image?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            # input_ids = tokenizer([caption_prompt], return_tensors='pt', truncation=True, max_length=4076)['input_ids']
            # attn_mask = input_ids != tokenizer.pad_token_id
            # input_ids = input_ids.to(llava.device)
            # attn_mask = attn_mask.to(llava.device)
            # outputs = llava.generate(input_ids, pixel_values=images, attention_mask=attn_mask, pad_token_id=tokenizer.pad_token_id, num_return_sequences=1, max_new_tokens=input_ids.shape[1]+20)
            # response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
            # print ('id', exp['id'])
            # print ('Caption', response)
            # print ()
            # continue
            if exp['question_type'] == 'multiple-choice':
                option_loss = []
                option_prompt = prompt# + f"The correct option is"
                #print (option_prompt)
                input_ids = tokenizer([option_prompt], return_tensors='pt', padding=True, truncation=True, max_length=4096)['input_ids']
                attn_mask = input_ids != tokenizer.pad_token_id
                input_ids = input_ids.to(llava.device)
                attn_mask = attn_mask.to(llava.device)
                #outputs = llava(input_ids, images, attention_mask=attn_mask)
                #output_ids = llava.generate(input_ids, pixel_values=images, attention_mask=attn_mask, pad_token_id=tokenizer.pad_token_id, do_sample=True, temperature=1, top_p=None, num_beams=5, max_new_tokens=64,use_cache=True)
                try:
                    output_ids = llava.generate(input_ids, pixel_values=images, attention_mask=attn_mask, pad_token_id=tokenizer.pad_token_id, max_new_tokens=64,use_cache=True)
                    input_token_len = input_ids.shape[1]
                    response = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=False)[0]
                    response = response.split('<|eot_id|>')[0]
                    chosen = parse_multi_choice_response(response, symbols[:len(options)], {s:o for s, o in zip(symbols[:len(options)], options)})
                    correct = eval_multi_choice(exp['answer'], chosen)
                except:
                    print (prompt)
                    print ('num images', len(images))
                    continue
                #print (pred)
                #exit()
                #try:
                # for option in symbols[:len(options)]:
                #     #print ('input_ids shape:', input_ids.shape)
                #     labels = tokenizer([option], return_tensors='pt', padding=True, truncation=True, max_length=4096)['input_ids']
                #     labels = labels.to(llava.device)
                #     #print ('labels shape:', labels.shape) 
                #     assert labels.shape[1] == 1               
                #     #print (outputs['logits'].shape)
                #     loss = loss_fct(outputs['logits'][:, -1, :].view(-1, outputs['logits'].shape[-1]), labels.view(-1))
                #     option_loss.append(loss.item())
                    #exit()
                # except:
                #     print (prompt)
                #     print (images.shape)
                #     print (exp['id'])
                #     exit()
                #     continue
                #chosen = option_loss.index(min(option_loss))
                #print ('chosen index', chosen, 'gold answer', exp['answer'])
            else:
                input_ids = tokenizer([prompt], return_tensors='pt', truncation=True, max_length=4076)['input_ids']
                attn_mask = input_ids != tokenizer.pad_token_id
                input_ids = input_ids.to(llava.device)
                attn_mask = attn_mask.to(llava.device)
                #print (prompt)
                # print (images.shape)
                output_ids = llava.generate(input_ids, pixel_values=images, attention_mask=attn_mask, pad_token_id=tokenizer.pad_token_id, max_new_tokens=64,use_cache=True)
                input_token_len = input_ids.shape[1]
                response = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=False)[0]
                response = response.split('<|eot_id|>')[0]
                chosen = parse_open_response(response)
                #print ('open-ended', response)
                #chosen = response.split('<|eot_id|>')[0]
                #print (type(chosen), type(exp['answer']), chosen, exp['answer'])
                correct = eval_open(exp['answer'], chosen)
            #print ('PRED', chosen, 'GOLD', exp['answer'])
            matches.append(correct)
            #exit()
            results.append({'question': question, 'options': options, 'chosen': chosen, 'gold': exp['answer']})
        print ('shard', args.shard, 'acc', sum(matches)/len(matches), 'correct', sum(matches), 'total', len(matches))
        with open(f'/path/to/weboutput_llava/checkpoint/llava-sft-llama3-8B-lr-2e-5-bs-2-seqlen-8192-pr-bf16-tp-8-pp-1-ac-full-do-true-sp-true-mantis_HF/MMMU_dev_results_{args.shard}.json', 'w') as fout:
            json.dump(results, fout)



def main(args):
    #if args.type == "pixel_values":
    #run_llava_pixel_values()
    # else:
    #     run_llava_image_features()
    run_llava_local_inference(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo on Llava")
    parser.add_argument("--type",
                        type=str,
                        choices=["pixel_values", "image_features"],
                        default="pixel_values",
                        help="image input type")
    parser.add_argument("--shard",
                        type=int,
                        default=0,
                        help="shard id")
    parser.add_argument("--num_shards",
                        type=int,
                        default=8,
                        help="number of shards")
    args = parser.parse_args()
    main(args)
