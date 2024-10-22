import os
import json
import shutil
import re 
import copy
from tqdm import tqdm

# eg, task dir ends with digit 3 or 12, remember that 12 > 3, but in alphabet, 3 > 12
# we can use zfill 00012 > 00003
def sort_key(tdir):

    if '--' in tdir:
        delim = '--'
    else:
        delim = '_'

    pre = delim.join( tdir.split(delim)[:-1] )
    post = tdir.split(delim)[-1].zfill(5)
    return pre + delim + post

def omit_history_last_n_steps(history, n_steps=3):
    new_history = []
    img_num = 0
    for turn_id in range(len(history)):
        curr_msg = history[len(history) - turn_id - 1]
        if curr_msg['from'] == 'gpt':
            new_history = [curr_msg] + new_history
        else:
            if img_num < n_steps-1:
                if curr_msg['value'].startswith("OBSERVATION:"):
                    curr_msg['value'] = 'OBSERVATION: <image> (accessibility tree omitted).'
                else:
                    curr_msg['value'] = curr_msg['value'].split("\nOBSERVATION:")[0] + '\nOBSERVATION: <image> (accessibility tree omitted).'
                img_num += 1
            else:
                if curr_msg['value'].startswith("OBSERVATION:"):
                    curr_msg['value'] = 'OBSERVATION: screenshot omitted, accessibility tree omitted.'
                else:
                    curr_msg['value'] = curr_msg['value'].split("\nOBSERVATION:")[0] + '\nOBSERVATION: screenshot omitted, accessibility tree omitted.'
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



def process_json_file(num_steps, flag_id, task_dir):
    json_file_path = os.path.join(task_dir, 'interact_messages.json')
    with open(json_file_path, 'r', encoding='utf-8') as fj:
        trajectory = json.load(fj)

    assert len(trajectory) == (num_steps * 2 + 1)

    generate_samples_full_context = []

    # webvoyager format
    task_info = trajectory[1]['content'][0]['text']
    assert 'Now given a task' in task_info
    pattern = r"Now given a task:(.+?)Please interact with"
    matches = re.search(pattern, task_info, re.DOTALL)
    task_content = matches.group(1).strip()

    for item_id in range(num_steps + 1):
        if item_id == 0:
            continue
        
        user_id = 2*item_id - 1
        assistant_id = 2*item_id
        item_user = trajectory[user_id]
        item_assistant = trajectory[assistant_id]

        assert item_user["role"] == 'user'
        assert item_assistant["role"] == 'assistant'

        last_n = 3
        if not generate_samples_full_context:
            new_item_full_context = []
        else:
            new_item_full_context = omit_history_last_n_steps(copy.deepcopy(generate_samples_full_context[-1]["conversations"]), last_n)
        
        acc_tree_path = os.path.join(task_dir, f'accessibility_tree{item_id}.txt')
        with open(acc_tree_path, 'r', encoding='utf-8') as facc:
            acc_tree = facc.read()
            # if acc_tree is too long.
            acc_tree = filter_acc_tree(acc_tree)
        
        # fail obs
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
            "from": "human",
            "value": f"{task_content}\nOBSERVATION:\n<image>\n{acc_tree}" if item_id == 1 else f"{fail_obs}OBSERVATION:\n<image>\n{acc_tree}"
        }

        current_turn_assistant = {
            "from": "gpt",
            "value": item_assistant["content"]
        }

        new_item_full_context.append(current_turn_user)
        new_item_full_context.append(current_turn_assistant)

        curr_flag_id = f"{flag_id}--step{str(item_id).zfill(2)}" 
        # curr_item = {
        #     "id": curr_flag_id,
        #     "image": f"{curr_flag_id}.png",
        #     "conversations": new_item_full_context
        # }
        image_list = []
        start_img_no = max(1, item_id-last_n+1)
        for img_id in range(start_img_no, item_id+1):
            image_list.append(f"{flag_id}--step{str(img_id).zfill(2)}.png" )

        curr_item = {
            "id": curr_flag_id,
            "images": image_list,
            "conversations": new_item_full_context
        }
        generate_samples_full_context.append(curr_item)
    

    assert len(generate_samples_full_context) == num_steps
    return generate_samples_full_context





def rename_and_move_images(flag_id, task_dir, image_dir):
    img_files = [ff for ff in os.listdir(task_dir) if ff.endswith('.png')]
    for img_file in img_files:
        img_id = img_file.split('screenshot')[1].split('.png')[0].zfill(2)
        new_img_file_name = f"{flag_id}--step{img_id}.png"
        if not os.path.exists(os.path.join(image_dir, new_img_file_name)):
            shutil.copy(os.path.join(task_dir, img_file), os.path.join(image_dir, new_img_file_name))
            print('move the image')
        assert os.path.exists(os.path.join(image_dir, new_img_file_name))
    return len(img_files)


# mind2web_select_round1, mind2web_select_round2, webvoyager_extend_round1, webvoyager_extend_round2,
# mind2web_human_extend_round1, mind2web_human_extend_round2, webvoyager_human_extend_round1, webvoyager_human_extend_round2

result_dir = "IL_trajectories/webvoyager_trajectories"
image_dir = "../OpenWebVoyager_image_data/webvoyager_images_IL"
save_json_file = "IL_json_data/webvoyager_json_data/example.json"
round_id = 1

os.makedirs(image_dir, exist_ok=True)
assert os.path.exists(image_dir)



task_dirs = os.listdir(result_dir)
task_dirs = sorted(task_dirs, key=sort_key)


finish_num = 0
data_item_num = 0

all_generate_samples_full_context = []
for dir_id in tqdm(range(len(task_dirs))):
    task_dir = task_dirs[dir_id]
    flag_id = sort_key(task_dir.replace('task', ''))

    task_dir = os.path.join(result_dir, task_dir)
    log_file = os.path.join(task_dir, 'agent.log')

    flag_id = f"{flag_id}--r{str(round_id).zfill(2)}"
    with open(log_file, 'r', encoding='utf-8') as fr1:
        log_content1 = fr1.read()
        if "INFO - finish!!" in log_content1:
            num_imgs = rename_and_move_images(flag_id, task_dir, image_dir)
            finish_num += 1
            generate_samples_full_context_3_img = process_json_file(num_imgs, flag_id, task_dir)
            data_item_num += len(generate_samples_full_context_3_img)
        else:
            continue
    
    all_generate_samples_full_context.extend(generate_samples_full_context_3_img)

# 209, 1578 for mind2web_round1
# 198, 1505 for mind2web_round2
# 199, 1314 for webvoyager_extend_round1
# 37, 320 for webvoyager_extend_round2
# 131, 691 for mind2web_human_extend_round1
# 10, 71 for mind2web_human_extend_round2
# 70, 370 for webvoyager_human_extend_round1
#  2, 17 for webvoyager_human_extend_round2
print(finish_num)
print(data_item_num)


with open(save_json_file, 'w', encoding='utf-8') as fw:
    json.dump(all_generate_samples_full_context, fw, indent=2)



