import re
import os
import time
import random
import copy
import io
import pickle
import base64
from PIL import Image
from typing import List, Dict, Any


# define a retry decorator
def retry_with_exponential_backoff(
        initial_delay: float = 1,
        exponential_base: float = 1.1,
        jitter: bool = True,
        max_retries: int = 40,
):
    """Retry a function with exponential backoff."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Initialize variables
            rate_limit_retry_num = 0
            delay = initial_delay

            # Loop until a successful response or max_retries is hit or an exception is raised
            while True:
                try:
                    return func(*args, **kwargs)
                # retry on all exceptions and errors
                except Exception as e:
                    print(f"Try count: {rate_limit_retry_num}, Error: {e}")
                    # Increment retries
                    rate_limit_retry_num += 1

                    # Check if max retries has been reached
                    if rate_limit_retry_num > max_retries:
                        raise Exception(
                            f"Maximum number of retries ({max_retries}) exceeded."
                        )

                    # Increment the delay
                    delay *= exponential_base * (1 + jitter * random.random())

                    # Sleep for the delay
                    print(f"Failure, sleep {delay} secs")
                    time.sleep(delay)
        return wrapper
    return decorator


# TODO: How to deal with PDF file?
def omit_history_last_n_steps(history, n_steps=3):
    new_history = []
    img_num = 0
    for turn_id in range(len(history)):
        curr_msg = history[len(history) - turn_id - 1]
        if curr_msg['role'] == 'assistant':
            new_history = [curr_msg] + new_history
        else:
            if img_num < n_steps-1:
                if curr_msg['content'].startswith("OBSERVATION:"):
                    curr_msg['content'] = 'OBSERVATION: <image> (accessibility tree omitted).'
                else:
                    curr_msg['content'] = curr_msg['content'].split("\nOBSERVATION:")[0] + '\nOBSERVATION: <image> (accessibility tree omitted).'
                img_num += 1
            else:
                if curr_msg['content'].startswith("OBSERVATION:"):
                    curr_msg['content'] = 'OBSERVATION: screenshot omitted, accessibility tree omitted.'
                else:
                    curr_msg['content'] = curr_msg['content'].split("\nOBSERVATION:")[0] + '\nOBSERVATION: screenshot omitted, accessibility tree omitted.'
            new_history = [curr_msg] + new_history
    return new_history

def filter_acc_tree(acc_tree):
    acc_tree_each_line = acc_tree.split('\n')
    acc_tree_each_line_new = []
    for line in acc_tree_each_line:
        if len(line.split(' ')) > 500:
            new_line = ' '.join(line.split()[:500])
            acc_tree_each_line_new.append(new_line)
        else:
            acc_tree_each_line_new.append(line)
    return '\n'.join(acc_tree_each_line_new)


def convert_msg_format_openai_to_localhost(message_obj: list[dict[str, Any]], task_dir, last_n = 3):
    assert len(message_obj) % 2 == 0

    task_info = message_obj[1]['content'][0]['text']
    assert 'Now given a task' in task_info
    pattern = r"Now given a task:(.+?)Please interact with"
    matches = re.search(pattern, task_info, re.DOTALL)
    task_content = matches.group(1).strip()

    img_files = [ff for ff in os.listdir(task_dir) if ff.endswith('.png')]
    num_steps = len(img_files)

    converted_sample = []
    for item_id in range(num_steps + 1):
        if item_id == 0:
            continue
        user_id = 2*item_id - 1
        item_user = message_obj[user_id]
        assert item_user["role"] == 'user'

        acc_tree_path = os.path.join(task_dir, f'accessibility_tree{item_id}.txt')
        with open(acc_tree_path, 'r', encoding='utf-8') as facc:
            acc_tree = facc.read()
            acc_tree = filter_acc_tree(acc_tree)
        
        fail_obs = ""
        fail_obs_1 = "The action you have chosen cannot be exected. Please double-check if you have selected the wrong Numerical Label or Action or Action format. Then provide the revised Thought and Action."
        fail_obs_2 = "The action you have chosen cannot be executed. Please double-check if you have selected the correct element or used correct action format. Then provide the revised Thought and Action."
        fail_obs_3 = "Format ERROR: Both 'Thought' and 'Action' should be included in your reply."

        user_msg = item_user["content"][0]["text"]
        if fail_obs_1 in user_msg or fail_obs_2 in user_msg:
            fail_obs = fail_obs_2 + '\n'
        elif fail_obs_3 in user_msg:
            fail_obs = fail_obs_3 + '\n'

        current_turn_user = {
            "role": "user",
            "content": f"{task_content}\nOBSERVATION:\n<image>\n{acc_tree}" if item_id == 1 else f"{fail_obs}OBSERVATION:\n<image>\n{acc_tree}"
        }

        if item_id != num_steps:
            assistant_id = 2*item_id
            item_assistant = message_obj[assistant_id]
            assert item_assistant["role"] == 'assistant'

            current_turn_assistant = {
                "role": "assistant",
                "content": item_assistant["content"]
            }
        
        if converted_sample:
            converted_sample = omit_history_last_n_steps(copy.deepcopy(converted_sample), last_n)
        

        converted_sample.append(current_turn_user)
        if item_id != num_steps:
            converted_sample.append(current_turn_assistant)
    
    image_list = []
    start_img_no = max(1, num_steps-last_n+1)
    for img_id in range(start_img_no, num_steps+1):
        image_list.append(os.path.join(task_dir, f"screenshot{img_id}.png"))
    curr_item = {
        "id": task_dir,
        "images": image_list,
        "conversations": converted_sample
    }

    # print(curr_item)

    example = {}
    example["id"] = curr_item["id"]
    img_list = []
    for curr_image_f in curr_item["images"]:
        with open(curr_image_f, "rb") as stream:
            image_bytes = stream.read()
            img_list.append(Image.open(io.BytesIO(image_bytes)))
    img_list = pickle.dumps(img_list)
    img_list_str = base64.b64encode(img_list).decode('utf-8')

    example["images"] = img_list_str
    example["conversations"] = curr_item["conversations"]
    
    return example