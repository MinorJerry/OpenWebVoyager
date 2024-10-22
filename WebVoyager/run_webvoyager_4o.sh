#!/bin/bash
nohup python -u run.py \
    --test_file ./data_for_training/IL/webvoyager_human_extend_75q.jsonl \
    --api_key YOUR_OPENAI_API_KEY \
    --headless \
    --max_iter 15 \
    --max_attached_imgs 3 \
    --temperature 1 \
    --seed 42 \
    --save_accessibility_tree \
    --api_model gpt-4o > webvoyager_human_extend_75q-gpt4o.log &