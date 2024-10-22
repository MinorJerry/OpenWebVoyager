#!/bin/bash
nohup python -u exploration_feedback.py \
    --test_file ./data_for_training/optim_iter1/webvoyager_for_iter1_150q.jsonl \
    --headless \
    --max_iter 15 \
    --max_attached_imgs 3 \
    --save_accessibility_tree \
    --api_eval_model gpt-4o \
    --max_rerun_times 5 \
    --api_localhost http://xxx.xxx.xxx.xxx:8080/predict > webvoyager_for_iter1_150q_rerun_5.log &
