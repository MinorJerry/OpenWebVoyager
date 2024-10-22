import os
import webdataset as wds
import glob
import json
import io
import pickle
from PIL import Image
from tqdm import tqdm

def convert_mantis_webdataset_for_webvoyager():
    # path to this folder
    abs_path = "."
    
    # json file path
    sub_dirs = glob.glob(f'{abs_path}/IL_json_data/*')
    json_data = []
    for data_dir in sub_dirs:
        if not os.path.isdir(data_dir):
            continue
        json_files = glob.glob(f'{data_dir}/*.json')
        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as fj:
                json_data.extend(json.load(fj))
            # print(json_file)
   
    # Images path
    abs_path_img = "../"
    sub_dirs = glob.glob(f"{abs_path_img}/OpenWebVoyager_image_data/*")
    all_images_files = []
    for image_dir in sub_dirs:
        image_files = glob.glob(f"{image_dir}/*.png")
        all_images_files.extend(image_files)
    
    # tar file path
    output_dir = f"{abs_path}/IL_tar_data"
    sink = wds.ShardWriter(f"{output_dir}/openwebvoyager-%06d.tar", maxcount=500, maxsize=3e10)

    for exp in tqdm(json_data):
        final_uid = exp["id"].replace('.','_')
        images_id = exp["images"] # eg: [Amazon--extend--00000--r01--step1.png, ...]
        img_list = []
        # read screenshots shown in: json_data images list
        for image_id in images_id:
            curr_image_f = ''
            # print(image_id)
            for image_f in all_images_files:
                if image_f.split('/')[-1] == image_id:
                    curr_image_f = image_f
                    break
            with open(curr_image_f, "rb") as stream:
                image_bytes = stream.read()
            img_list.append(io.BytesIO(image_bytes))
        img_list = pickle.dumps(img_list)
        sink.write({  "__key__": final_uid,
              "input_image": img_list,
               "conversations": str(list(exp['conversations'])),})
    sink.close()
    shardlist = []
    shard = 0
    for start in range(0, len(json_data), 500):
        shardlist.append({'url': f"openwebvoyager-{shard:06d}.tar", 'nsamples': min(500, len(json_data) - start)})
        shard += 1
    with open(f'{output_dir}/shardlist.json', 'w') as fout:
        json.dump(shardlist, fout, indent=4)

if __name__ == '__main__':
    convert_mantis_webdataset_for_webvoyager()