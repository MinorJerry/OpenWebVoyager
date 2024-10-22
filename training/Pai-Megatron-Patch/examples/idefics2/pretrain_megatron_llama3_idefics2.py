# Copyright (c) 2023 Alibaba PAI and Nvidia Megatron-LM Team.
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

from functools import partial
import torch
import pdb

from megatron.core.enums import ModelType
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training import get_args
from megatron.training import get_timers
from megatron.core import tensor_parallel
from megatron.training.utils import average_losses_across_data_parallel_group
from megatron.legacy.data.data_samplers import MegatronPretrainingRandomSampler
from megatron.training.training import cyclic_iter
from megatron.core import mpu
from megatron.training.utils import get_ltor_masks_and_position_ids

from megatron_patch.data import build_pretrain_dataset_from_original
from megatron_patch.model.idefics2.gpt_model import GPTModel
from megatron_patch.tokenizer import build_tokenizer
from megatron_patch.tokenizer import get_tokenizer
from megatron_patch.training import pretrain
from megatron_patch.arguments import get_patch_args
from megatron_patch.data.idefics2.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, PAD_IDX, AROUND_IMAGE_TOKEN_INDEX, RELETIVE_IMAGE_TOKEN_ID

def model_provider(pre_process=True, post_process=True):
    args = get_args()
    build_tokenizer(args)
    print ('Building Idefics2 model, preprocess =', pre_process, 'postprocess =', post_process)
    config = core_transformer_config_from_args(get_args())
    config.variable_seq_lengths = True
    model = GPTModel(
        config,
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process
    )
    return model


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    text_keys = ['input_ids', 'labels']
    text_attention_keys = ['attention_mask']
    img_keys = ['image']
    img_mask_keys = ['pixel_attention_mask']
    # pdb.set_trace()

    if data_iterator is not None:
        data = next(data_iterator)
        while data is None:
            data = next(data_iterator)
    else:
        data = None
    #data_text = {'input_ids': data['input_ids'], 'labels': data['labels']}
    #data_image = {'image': data['image']}
    data_text = tensor_parallel.broadcast_data(text_keys, data, torch.int64)
    if args.fp16:
        dtype = torch.half
    elif args.bf16:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    data_image = tensor_parallel.broadcast_data(img_keys, data, dtype)
    image_attention_masks = tensor_parallel.broadcast_data(img_mask_keys, data, dtype)[img_mask_keys[0]]
    tokens = data_text['input_ids'].long()
    labels = data_text['labels'].long()
    images = data_image[img_keys[0]]

    data_b1 = tensor_parallel.broadcast_data(['weights'], data, torch.float32)
    loss_mask = data_b1['weights']
    #print (labels[:, :20])
    #print ('tokens', tokens.shape, 'labels', labels.shape, 'images', images.shape)
    # tokens = tokens_[:, :-1].contiguous()
    # labels = labels_[:, 1:].contiguous()
    # attention_mask = tokens.ne(IGNORE_INDEX)
    # attention_mask = tokens.ne(IGNORE_INDEX)
    #attention_mask = tensor_parallel.broadcast_data(text_attention_keys, data, torch.int64)[text_attention_keys[0]]
    #answer_mask = tensor_parallel.broadcast_data(['answer_mask'], data, torch.int64)['answer_mask'] # mask for answers
    # Get the masks and postition ids.
    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        labels,
        tokenizer.eos_token_id,
        args.reset_position_ids,
        args.reset_attention_mask,
        True)
    # loss_mask = torch.ones(tokens.size(), dtype=torch.float, device=tokens.device)
    # if eod_mask_loss:
    # loss_mask[tokens == PAD_IDX] = 0.0 # for unk (padding tokens), don't calc loss.
    #loss_mask[tokens == IMAGE_TOKEN_INDEX] = 0.0
    #loss_mask[tokens == 0] = 0.0 # unk
    
    # loss_mask[tokens == AROUND_IMAGE_TOKEN_INDEX] = 0.0
    
    # for idefics
    #position_ids = attention_mask.long().cumsum(-1) - 1
    #position_ids.masked_fill_(attention_mask == 0, 0)
    
    #if not args.answer_loss_only:
    #    answer_mask = loss_mask
        
    return tokens, labels, loss_mask, attention_mask, position_ids, images, image_attention_masks

def loss_func(loss_mask, labels, pad_vocab_size, output_tensor):
    # if torch.distributed.get_rank() == 0 :
    #     torch.save(loss_mask, "debug_output/loss_mask.pt")
    #     torch.save(answer_mask, "debug_output/answer_mask.pt")

    # loss_mask = loss_mask * answer_mask

    # WARNING: FTQ: may have error for pipeline parallellism, not sure
    # mask out pad_vocab_size for the last gpu

    # if torch.distributed.get_rank() == torch.distributed.get_world_size() - 1: 
    #     output_tensor[:, :, -pad_vocab_size:] = -100
    #     output_tensor[:, :, -(pad_vocab_size + RELETIVE_IMAGE_TOKEN_ID)] = -100 # this is the ignore_index when calculating cross entropy.


    logits = output_tensor[:, :-1, :].contiguous()
    labels = labels[:, 1:].contiguous()
    loss_mask = loss_mask[:, 1:].contiguous()
    # need to mask out in advance.
    losses = tensor_parallel.vocab_parallel_cross_entropy(
            logits.contiguous().float(), labels.contiguous())
    loss_mask = loss_mask.view(-1).float()
    # if torch.distributed.get_rank() == 0 :
    #     torch.save(losses, "debug_output/losses.pt")
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    if mpu.get_tensor_model_parallel_rank() == 0 and loss.isnan():
        # print ('logits', logits.shape, 'labels', labels.shape, 'loss_mask', loss_mask.shape)
        print ('logits', logits.shape, logits[:, :10, :10])
        print (labels)
        #print (loss_mask)
        #print (losses)
        print (losses.view(-1) * loss_mask,  loss_mask.sum())
        print (loss)
        exit()
    #exit()
    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])
    return loss, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    timers = get_timers()
    args = get_args()
    pad_vocab_size = args.pad_vocab_size
    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids, images, image_attention_mask = get_batch(
        data_iterator)

    timers('batch-generator').stop()

    # if torch.distributed.get_rank() == 0:
    #     torch.save({"input_ids":tokens, "attention_mask":attention_mask, "position_ids":position_ids, "pixel_values":images, "pixel_attention_mask":image_attention_mask}, "test_batch.pt")

    output_tensor = model(tokens, position_ids, attention_mask,
                          images=images, pixel_attention_mask=image_attention_mask)

    # return output_tensor, partial(loss_func, total_loss_mask, total_label)
    return output_tensor, partial(loss_func, loss_mask, labels, pad_vocab_size)
    # TODO: mod loss_mask 


def mm_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        print ('batch empty')
        return None
    max_len = max([len(item['input_ids']) for item in batch])
    while max_len % 8 != 0:
        max_len += 1
    input_ids = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)
    labels = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)
    weights = torch.zeros(len(batch), max_len, dtype=torch.float)
    preferences = torch.zeros(len(batch), max_len, dtype=torch.float)
    for i, item in enumerate(batch):
        input_ids[i, :len(item['input_ids'])] = item['input_ids']
        labels[i, :len(item['labels'])] = item['labels']
        weights[i, :len(item['weights'])] = item['weights']
        preferences[i, :len(item['preferences'])] = item['preferences']
    # input_ids = torch.stack([item['input_ids'] for item in batch])
    # labels = torch.stack([item['labels'] for item in batch])
    # weights = torch.stack([item['weights'] for item in batch])
    # preferences = torch.stack([item['preferences'] for item in batch])
    batch_images = [item['image'] for item in batch if 'image' in item and item['image'] is not None]
    batch_pixal_masks = [item['pixel_attention_mask'] for item in batch if 'pixel_attention_mask' in item and item['pixel_attention_mask'] is not None]
    if len(batch_images) > 0:
        images = torch.cat(batch_images, dim=0)
        pixel_masks = torch.cat(batch_pixal_masks, dim=0)
    else:
        images = torch.zeros(1, 10, dtype=torch.bfloat16)
        pixel_masks = torch.zeros(1, 10, dtype=torch.bfloat16)
    #print ('batch images', images.shape, 'pixel masks', pixel_masks.shape)
    return {'input_ids': input_ids, 'labels': labels, 'weights': weights, 'preferences': preferences, 'image': images, 'pixel_attention_mask': pixel_masks}

def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()
    train_ds, valid_ds, test_ds = \
        build_pretrain_dataset_from_original(args.dataset)
    if args.dataloader_type in ['cyclic', 'single']:
        return train_ds, valid_ds, test_ds
    batch_sampler = MegatronPretrainingRandomSampler(
            train_ds,
            total_samples=len(train_ds),
            consumed_samples=0,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size(),
            data_sharding=args.data_sharding)
    print ('Current data parallel rank is', mpu.get_data_parallel_rank(), 'world size', mpu.get_data_parallel_world_size())
    train_loader = torch.utils.data.DataLoader(train_ds,
                                        batch_sampler=batch_sampler,
                                        num_workers=args.num_workers,
                                        pin_memory=True,
                                        persistent_workers=True if args.num_workers > 0 else False,
                                        collate_fn = mm_collate_fn
                                        )
    train_iterator = iter(cyclic_iter(train_loader))
    return train_iterator, train_iterator, train_iterator

if __name__ == "__main__":

    pretrain(train_valid_test_datasets_provider, model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             extra_args_provider=get_patch_args)