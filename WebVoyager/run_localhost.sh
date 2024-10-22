#!/bin/bash
nohup python -u run.py \
    --test_file ./data_for_test/example.jsonl \
    --headless \
    --max_iter 15 \
    --max_attached_imgs 3 \
    --save_accessibility_tree \
    --output_dir ./results_example_test \
    --api_localhost http://xxx.xxx.xxx.xxx:8080/predict > results_example_test.log &
