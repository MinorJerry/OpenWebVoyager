#!/bin/bash
# megatron to transformers: You need to copy the tokenizer files into the save_path
# bash model_convertor.sh ../../Megatron-LM/ ../../llama-hf2mg-test-2-2/release/ ../../llama_mg2hf 1 1 llama-7b 1 true
# transformers to megatron
# bash model_convertor.sh ../../Megatron-LM/ ../../llama-7b-hf ../../llama-hf2mg 1 1 llama-7b 1 false
set -e
START_TIME=$SECONDS

MEGATRON_PATCH_PATH=/path/to/Pai-Megatron-Patch/
MEGATRON_PATH=${MEGATRON_PATCH_PATH}/Megatron-LM-240424

MODEL_PATH=OpenWebVoyager-img3-idefics-7B-lr-1e-5-bs-1-seqlen-8192-pr-bf16-tp-8-pp-1-ac-full-do-false-sp-false-warmup-9

SOURCE_CKPT_PATH=/path/to/checkpoint/${MODEL_PATH}/iter_0000300
CONFIG_PATH=/path/to/idefics2-8b-config-folder


TP=8
PP=1
MN=mistral-7b
EXTRA_VOCAB_SIZE=125
mg2hf=true
dtype=bf16

if [ $mg2hf = true ]; then
    do_options="
                --convert_checkpoint_from_megatron_to_transformers
    "
elif [ $mg2hf = false ]; then
    do_options=""
fi

export PYTHONPATH=${MEGATRON_PATH}:$PYTHONPATH

TARGET_CKPT_PATH=/path/to/${MODEL_PATH}/hf_ckp
echo $TARGET_CKPT_PATH


python idefics2_hf2mg.py \
--load_path ${SOURCE_CKPT_PATH} \
--config_path ${CONFIG_PATH} \
--save_path ${TARGET_CKPT_PATH} \
--target_params_dtype ${dtype} \
--megatron-path ${MEGATRON_PATH} \
--target_tensor_model_parallel_size ${TP} \
--target_pipeline_model_parallel_size ${PP} \
--model_name ${MN} \
--extra_num_vocabs ${EXTRA_VOCAB_SIZE} \
${do_options}

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"
