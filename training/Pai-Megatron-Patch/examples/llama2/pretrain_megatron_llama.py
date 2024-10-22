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
import os

from megatron.core.enums import ModelType
from megatron.training.utils import get_ltor_masks_and_position_ids
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training import get_args
from megatron.training import get_timers

from megatron.core import tensor_parallel
from megatron.training.utils import average_losses_across_data_parallel_group
from megatron.legacy.data.data_samplers import MegatronPretrainingRandomSampler
from megatron.training.training import cyclic_iter
from megatron.core import mpu

from megatron_patch.data import \
    build_pretrain_dataset_from_original, build_pretrain_dataset_from_idxmap
from megatron_patch.model.llama2.gpt_model import GPTModel
from megatron_patch.tokenizer import get_tokenizer, build_tokenizer
from megatron_patch.training import pretrain
from megatron_patch.arguments import get_patch_args
IGNORE_INDEX = None

def model_provider(pre_process=True, post_process=True):
    args = get_args()
    build_tokenizer(args)
    global IGNORE_INDEX
    IGNORE_INDEX = get_tokenizer().pad_token_id
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
    datatype = torch.int64

    keys = ['input_ids', 'labels']
    if data_iterator is not None:
        data = next(data_iterator)
        if data['input_ids'].dtype != torch.int64:
            print ('data type:', data['input_ids'].dtype, 'converting to long')
            data['input_ids'] = data['input_ids'].long()
            data['labels'] = data['labels'].long()
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)
    tokens = data_b['input_ids'].long()
    labels = data_b['labels'].long()
    data_b1 = tensor_parallel.broadcast_data(['weights'], data, torch.float32)
    loss_mask = data_b1['weights']
    # labels = tokens_[:, 1:].contiguous()
    # tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        labels,
        tokenizer.eos_token_id,
        args.reset_position_ids,
        args.reset_attention_mask,
        True)
    return tokens, labels, loss_mask, attention_mask, position_ids

def loss_func(loss_mask, labels, output_tensor):
    # losses = output_tensor.float()
    # loss_mask = loss_mask.view(-1).float()
    #print ('output_tensor', output_tensor.shape, 'labels', labels.shape, 'loss_mask', loss_mask.shape)
    logits = output_tensor[:, :-1, :].contiguous()
    labels = labels[:, 1:].contiguous()
    loss_mask = loss_mask[:, 1:].contiguous()
    # print ('loss_mask', loss_mask.shape, loss_mask.sum(dim=1))
    losses = tensor_parallel.vocab_parallel_cross_entropy(
            logits.contiguous().float(), labels.contiguous())
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    timers('batch-generator').stop()

    output_tensor = model(tokens, position_ids, attention_mask)

    return output_tensor, partial(loss_func, loss_mask, labels)

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
    preferences = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, item in enumerate(batch):
        input_ids[i, :len(item['input_ids'])] = torch.tensor(item['input_ids'], dtype=torch.long)
        labels[i, :len(item['labels'])] = torch.tensor(item['labels'], dtype=torch.long)
        weights[i, :len(item['weights'])] = torch.tensor(item['weights'], dtype=torch.float)
        preferences[i, :len(item['preferences'])] = torch.tensor(item['preferences'], dtype=torch.long)
    return {'input_ids': input_ids, 'labels': labels, 'weights': weights, 'preferences': preferences}

def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    if os.path.isfile(args.train_data_path[0]):
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
        train_loader = torch.utils.data.DataLoader(train_ds,
                                            batch_sampler=batch_sampler,
                                            num_workers=args.num_workers,
                                            pin_memory=True,
                                            persistent_workers=True if args.num_workers > 0 else False,
                                            collate_fn = mm_collate_fn
                                            )
        train_iterator = iter(cyclic_iter(train_loader))
        return train_iterator, train_iterator, train_iterator
    else:
        train_ds, valid_ds, test_ds = \
            build_pretrain_dataset_from_idxmap(
                data_prefix=args.train_data_path,
                max_padding_length=args.max_padding_length,
                dataset_type=args.dataset,
                splits_string=args.split,
                train_valid_test_num_samples=train_val_test_num_samples,
                seed=args.seed,
                skip_warmup=(not args.mmap_warmup)
            )

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    pretrain(train_valid_test_datasets_provider, model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             extra_args_provider=get_patch_args)
