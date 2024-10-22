import os
import subprocess
from multiprocessing import Pool
import glob
from tqdm import tqdm
import json 
from PIL import Image
from collections import Counter 
import webdataset as wds
import random 
import zipfile
import pyarrow.parquet as pq
import pandas as pd
import io
import pickle

# change the keys in the conversations:
def rename_dict(d):
    new_d = {}
    role_dict = {'human':'user', 'gpt':'assistant'}
    new_d['role'] = role_dict[d['from']]
    new_d['content'] = d['value']
    return new_d

def rsync_folder(folder):
    """
    Synchronizes a single folder using rsync.
    
    :param folder: A tuple containing the source folder path and the destination folder path.
    """
    print (f"working on {folder}")
    source, destination = folder
    destination = '/'.join(destination.split('/')[:-1])
    # Construct the rsync command
    command = ["rsync", "-az", source, destination]
    # Execute the rsync command
    subprocess.run(command, check=True)
    print(f"Completed: {source}")

def parallel_rsync(source_base_folder, dest_base_folder, num_processes=4):
    """
    Copies each folder from the source to the destination in parallel using rsync.
    
    :param source_base_folder: Base folder containing subfolders to copy.
    :param dest_base_folder: Base destination folder where subfolders will be copied.
    :param num_processes: Number of parallel processes to use.
    """
    # List all folders in the source directory
    source_folders = sorted(glob.glob(os.path.join(source_base_folder, '00*')))
    dest_folders = sorted(glob.glob(os.path.join(dest_base_folder, '00*')))
    assert len(source_folders) == len(dest_folders), "Source and destination folders do not match"
    print ('number of source folders:', len(source_folders))
    todo = []
    folders = list(zip(source_folders, dest_folders))
    for src, dest in tqdm(folders):
        num_src = len(glob.glob(os.path.join(src, '*')))
        num_dest = len(glob.glob(os.path.join(dest, '*')))
        if num_src != num_dest:
            todo.append((src, dest))
    print(f"Number of folders to sync: {len(todo)}")
    print (todo[0])
    #rsync_folder(todo[0])
    # folders = [(os.path.join(source_base_folder, f), os.path.join(dest_base_folder, f))
    #            for f in os.listdir(source_base_folder)
    #            if os.path.isdir(os.path.join(source_base_folder, f))]
    
    # Create a pool of workers
    with Pool(processes=num_processes) as pool:
        pool.map(rsync_folder, todo)

def bulk_move():
    # Example usage
    source_base_folder = '/data1/pretrain_images/'
    dest_base_folder = '/path/to/LLaVA-Pretrain/pretrain_images/'
    num_processes = 10  # Adjust based on your system's capabilities

    parallel_rsync(source_base_folder, dest_base_folder, num_processes)

def prepare_webdataset_pretrain():
    random.seed(42)
    instruction_data = json.load(open('/data1/LLAVA_pretrain_images/blip_laion_cc_sbu_558k.json'))
    print (len(instruction_data))
    random.shuffle(instruction_data)
    seed42_data = []
    id_count = Counter()
    for exp in instruction_data:
        assert 'image' in exp
            # seed42_data.append(exp)
            # continue
        exp['conversations'] = [rename_dict(d) for d in exp['conversations']]
        id_count[exp['id']] += 1
        image_path = f'/data1/LLAVA_pretrain_images/{exp["image"]}'
        if os.path.exists(image_path):
            seed42_data.append(exp)
    print (len(seed42_data), len(id_count))
  
    sink = wds.ShardWriter(f"/data1/LLAVA_pretrain_images/llava_pretrain_data-%06d.tar", maxcount=83000, maxsize=3e10)
    print (len(instruction_data))
    print (instruction_data[500000])
    # image_found = 0
    # no_image = 0
    # missing_count = Counter()
    # id_count = Counter()
    # image_count = Counter()
    for exp in tqdm(instruction_data):
        # if "image" not in exp:
        #     no_image += 1
        #     # print (exp['id'])
        #     # exit()
        #     assert '/' not in exp['id']
        #     # fake_image = Image.new('RGB', (224, 224), color = 'red')
        #     # fake_image.save('/data1/LLAVA_finetune_images/fake_image.jpg')
        #     # exit()
        #     print (exp['id'])
        #     id_count[exp['id']] += 1
        #     continue
        #     with open('/data1/LLAVA_finetune_images/fake_image.jpg', "rb") as stream:
        #         fake_image = stream.read()
        #     sink.write({
        #             "__key__": exp["id"],
        #             "input_image": fake_image,
        #             "conversations": str(exp['conversations']),
        #         })
        #     continue
        # else:
        #     assert '/' in exp['image']
        image_path = f'/data1/LLAVA_pretrain_images/{exp["image"]}'
        #if os.path.exists(image_path):
        #image_found += 1
        image_dir = exp['image'].split('/')[0]
        # image_count[image_dir] += 1
        # id_count[f"{image_dir}_{image_count[image_dir]}"] += 1
        # continue 
        img = Image.open(image_path).convert('RGB')
        with open(image_path, "rb") as stream:
            img = stream.read()
        sink.write({
                "__key__": exp['id'],
                "input_image": img,
                "conversations": str(exp['conversations']),
            })
            
        # else:
        #     continue
        #     dir_name = exp["image"].split('/')[0]
        #     missing_count[dir_name] += 1
            # sink.write({
            #     "__key__": exp["id"],
            #     "input_image": None,
            #     "conversations": str(exp['conversations']),
            # })

    sink.close()
    # print (f"No image: {no_image}")
    # print (f"Image found: {image_found}", len(instruction_data))
    # print (f"Missing count: {missing_count}")
    # print (len(id_count))
    shardlist = []
    shard = 0
    for start in range(0, len(id_count), 83000):
        shardlist.append({'url': f"llava_pretrain_data-{shard:06d}.tar", 'nsamples': min(83000, len(id_count) - start)})
        shard += 1
    with open('/path/to/pretrain_data/LLaVA-Pretrain/shardlist.json', 'w') as fout:
        json.dump(shardlist, fout, indent=4)




def prepare_llava_instruct_new():
    random.seed(42)

    sink = wds.ShardWriter(f"/data1/LLAVA_finetune_images/llava_v1_5_data/new/llava_v1_5_data-%06d.tar", maxcount=25000, maxsize=3e10)
    # sink = wds.ShardWriter(f"/data1/LLAVA_finetune_images/llava_v1_5_data/tq_test/llava_v1_5_data-%06d.tar", maxcount=25000, maxsize=3e10)
    instruction_data = json.load(open('/data1/LLAVA_finetune_images/llava_v1_5_mix665k.json'))
    print("num original data", len(instruction_data))
    random.shuffle(instruction_data)
    seed42_data = []
    for i, exp in enumerate(instruction_data):
        # change the names of the keys
        exp['conversations'] = [rename_dict(d) for d in exp['conversations']]
        if 'image' not in exp:
            seed42_data.append(exp)
            continue
        image_path = f'/data1/LLAVA_finetune_images/{exp["image"]}'
        if os.path.exists(image_path):
            seed42_data.append(exp)
        
        # if i == 1000:
        #     break

    print("number of data after checking image paths", len(seed42_data))

    with open('/path/to/pretrain_data/LLaVA-Instruct-150K/new/llava_v1_5_mix665k_seed42.json', 'w') as fout:
        json.dump(seed42_data, fout)
    
    instruction_data = seed42_data
    # exit()
    
    print (len(instruction_data))
    # print (instruction_data[500000])
    image_found = 0
    no_image = 0
    id_count = Counter()
    image_count = Counter()
    for exp in tqdm(instruction_data):
        if "image" not in exp:
            no_image += 1
            assert '/' not in exp['id']
            # print (exp['id'])
            id_count[exp['id']] += 1
            # continue
            sink.write({
                    "__key__": exp["id"],
                    "conversations": str(exp['conversations']),
                })
        else:
            assert '/' in exp['image']
            image_path = f'/data1/LLAVA_finetune_images/{exp["image"]}'
            assert os.path.exists(image_path)
            image_found += 1
            image_dir = exp['image'].split('/')[0]
            image_count[image_dir] += 1
            id_count[f"{image_dir}_{image_count[image_dir]}"] += 1
            # img = Image.open(image_path).convert('RGB')
            with open(image_path, "rb") as stream:
                img = stream.read()
            sink.write({
                    "__key__": f"{image_dir}_{image_count[image_dir]}",
                    "input_image": img,
                    "conversations": str(exp['conversations']),
                })
            
        # else:
        #     continue
        #     dir_name = exp["image"].split('/')[0]
        #     missing_count[dir_name] += 1
            # sink.write({
            #     "__key__": exp["id"],
            #     "input_image": None,
            #     "conversations": str(exp['conversations']),
            # })

    sink.close()
    print (f"No image: {no_image}")
    print (f"Total num instances: {image_found}", len(instruction_data))
    # print (f"Missing count: {missing_count}")
    print (len(id_count))
    shardlist = []
    shard = 0
    for start in range(0, len(id_count), 25000):
        shardlist.append({'url': f"llava_v1_5_data-{shard:06d}.tar", 'nsamples': min(25000, len(id_count) - start)})
        shard += 1
    with open('/data1/LLAVA_finetune_images/llava_v1_5_data/new/shardlist.json', 'w') as fout:
    # with open('/data1/LLAVA_finetune_images/llava_v1_5_data/tq_test/shardlist.json', 'w') as fout:
        json.dump(shardlist, fout, indent=4)


def prepare_llava_pretrain_new():
    random.seed(42)
    instruction_data = json.load(open('/data1/LLAVA_pretrain_images/blip_laion_cc_sbu_558k.json'))
    print (len(instruction_data))
    random.shuffle(instruction_data)
    seed42_data = []
    id_count = Counter()
    for exp in instruction_data:
        assert 'image' in exp
            # seed42_data.append(exp)
            # continue
        id_count[exp['id']] += 1
        image_path = f'/data1/LLAVA_pretrain_images/{exp["image"]}'
        if os.path.exists(image_path):
            seed42_data.append(exp)
    print (len(seed42_data), len(id_count))

    sink = wds.ShardWriter(f"/data1/LLAVA_finetune_images/llava_pretrain_new/llava_pretrain_data-%06d.tar", maxcount=25000, maxsize=3e10)

    print("number of data after checking image paths", len(seed42_data))

    with open('/data1/LLAVA_finetune_images/llava_pretrain_new/llava_pretrain_seed42.json', 'w') as fout:
        json.dump(seed42_data, fout)
    
    instruction_data = seed42_data
    # exit()
    
    print (len(instruction_data))
    # print (instruction_data[500000])
    image_found = 0
    no_image = 0
    id_count = Counter()
    image_count = Counter()
    for exp in tqdm(instruction_data):
        if "image" not in exp:
            no_image += 1
            assert '/' not in exp['id']
            # print (exp['id'])
            id_count[exp['id']] += 1
            # continue
            sink.write({
                    "__key__": exp["id"],
                    "conversations": str(exp['conversations']),
                })
        else:
            assert '/' in exp['image']
            image_path = f'/data1/LLAVA_pretrain_images/{exp["image"]}'
            assert os.path.exists(image_path)
            image_found += 1
            image_dir = exp['image'].split('/')[0]
            image_count[image_dir] += 1
            id_count[f"{image_dir}_{image_count[image_dir]}"] += 1
            # img = Image.open(image_path).convert('RGB')
            with open(image_path, "rb") as stream:
                img = stream.read()
            sink.write({
                    "__key__": f"{image_dir}_{image_count[image_dir]}",
                    "input_image": img,
                    "conversations": str(exp['conversations']),
                })

    sink.close()
    print (f"No image: {no_image}")
    print (f"Total num instances: {image_found}", len(instruction_data))
    # print (f"Missing count: {missing_count}")
    print (len(id_count))
    shardlist = []
    shard = 0
    for start in range(0, len(id_count), 25000):
        shardlist.append({'url': f"llava_v1_5_data-{shard:06d}.tar", 'nsamples': min(25000, len(id_count) - start)})
        shard += 1
    with open('/data1/LLAVA_finetune_images/llava_pretrain_new/shardlist.json', 'w') as fout:
    # with open('/data1/LLAVA_finetune_images/llava_v1_5_data/tq_test/shardlist.json', 'w') as fout:
        json.dump(shardlist, fout, indent=4)


def prepare_webdataset():
    random.seed(42)

    #sink = wds.ShardWriter(f"/data1/LLAVA_finetune_images/llava_v1_5_data/llava_v1_5_data-%06d.tar", maxcount=83000, maxsize=3e10)
    instruction_data = json.load(open('/data1/LLAVA_finetune_images/llava_v1_5_mix665k.json'))
    random.shuffle(instruction_data)
    seed42_data = []
    for exp in instruction_data:
        if 'image' not in exp:
            seed42_data.append(exp)
            continue
        image_path = f'/data1/LLAVA_finetune_images/{exp["image"]}'
        if os.path.exists(image_path):
            seed42_data.append(exp)
    print (len(seed42_data))
    with open('/path/to/llava_v1_5_mix665k_seed42.json', 'w') as fout:
        json.dump(seed42_data, fout)
    exit()
    print (len(instruction_data))
    print (instruction_data[500000])
    image_found = 0
    no_image = 0
    missing_count = Counter()
    id_count = Counter()
    image_count = Counter()
    for exp in tqdm(instruction_data):
        if "image" not in exp:
            no_image += 1
            # print (exp['id'])
            # exit()
            assert '/' not in exp['id']
            # fake_image = Image.new('RGB', (224, 224), color = 'red')
            # fake_image.save('/data1/LLAVA_finetune_images/fake_image.jpg')
            # exit()
            print (exp['id'])
            id_count[exp['id']] += 1
            continue
            with open('/data1/LLAVA_finetune_images/fake_image.jpg', "rb") as stream:
                fake_image = stream.read()
            sink.write({
                    "__key__": exp["id"],
                    "input_image": fake_image,
                    "conversations": str(exp['conversations']),
                })
            continue
        else:
            assert '/' in exp['image']
        image_path = f'/data1/LLAVA_finetune_images/{exp["image"]}'
        if os.path.exists(image_path):
            image_found += 1
            image_dir = exp['image'].split('/')[0]
            image_count[image_dir] += 1
            id_count[f"{image_dir}_{image_count[image_dir]}"] += 1
            continue 
            img = Image.open(image_path).convert('RGB')
            with open(image_path, "rb") as stream:
                img = stream.read()
            sink.write({
                    "__key__": f"{image_dir}_{image_count[image_dir]}",
                    "input_image": img,
                    "conversations": str(exp['conversations']),
                })
            
        else:
            continue
            dir_name = exp["image"].split('/')[0]
            missing_count[dir_name] += 1
            # sink.write({
            #     "__key__": exp["id"],
            #     "input_image": None,
            #     "conversations": str(exp['conversations']),
            # })

    sink.close()
    print (f"No image: {no_image}")
    print (f"Image found: {image_found}", len(instruction_data))
    print (f"Missing count: {missing_count}")
    print (len(id_count))
    shardlist = []
    shard = 0
    for start in range(0, len(id_count), 83000):
        shardlist.append({'url': f"llava_v1_5_data-{shard:06d}.tar", 'nsamples': min(83000, len(id_count) - start)})
        shard += 1
    with open('/path/to/nlp/pretrain_data/LLaVA-Instruct-150K/llava_v1_5_data/shardlist.json', 'w') as fout:
        json.dump(shardlist, fout, indent=4)

def convert_mantis_webdataset():
    
    sub_dirs = glob.glob('/data1/LLAVA_finetune_images/TIGER-Lab/Mantis-Instruct/*')
    json_data = []
    uids = {}
    uid_count = Counter()
    for dir in sub_dirs:
        if not os.path.isdir(dir):
            continue
        # if 'star' not in dir:
        #     continue
        print (dir)
        dir_name = dir.split('/')[-1]
        parquet_files = glob.glob(f'{dir}/train*.parquet')
        dir_data = []
        for parquet_file in parquet_files:
            table = pq.read_table(parquet_file)
            df = table.to_pandas()
            list_of_dicts = df.to_dict('records')
            dir_data.extend(list_of_dicts)
        # zip_files = glob.glob(f'{dir}/train*.zip')
        # img_dict = {}
        # for zip_file in zip_files:
        #     with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        #         all_files = zip_ref.namelist()
        #         for i, img_files in enumerate(all_files):
        #             img = zip_ref.read(img_files)
        #             img = io.BytesIO(img)
        #             pkl_img = pickle.dumps([img, img, img])
        #             print (type(pkl_img))
        #             exit()
        #             if img_files.startswith('TJZ0P'):
        #                 img = Image.open(img).convert('RGB')
        #                 img.save(f'{img_files}')
        for exp in dir_data:
            exp['image'] = dir 
            exp['images'] = list(exp['images'])
            exp['conversation'] = list(exp['conversation'])
            json_data.append(exp)
            # if f"{dir_name}_{exp['id']}" == "lrv_multi_4224":
            #     print (exp)
            # if f"{dir_name}_{exp['id']}" in uids:
            #     print (f"{dir_name}_{exp['id']}")
            #     exit()
            #uid = f"{dir_name}_{exp['id']}"
            uid_count[dir_name] += 1
            uids[f"{dir_name}_{uid_count[dir_name]}"] = 1
        #     if len(json_data) == 25000:
        #         break
        # if len(json_data) == 25000:
        #     break
        #print (dir_data[:3])
    print (len(json_data), len(uids))
    

    uids = {}
    uid_count = Counter()
    sub_dirs = glob.glob('/data1/LLAVA_finetune_images/TIGER-Lab/Mantis-Instruct/*')
    sink = wds.ShardWriter(f"/data1/LLAVA_finetune_images/TIGER-Lab/mantis-instruct-%06d.tar", maxcount=25000, maxsize=3e10)
    for dir in sub_dirs:
        if not os.path.isdir(dir):
            continue
        print (dir)
        dir_name = dir.split('/')[-1]
        parquet_files = glob.glob(f'{dir}/train*.parquet')
        dir_data = []
        for parquet_file in parquet_files:
            table = pq.read_table(parquet_file)
            df = table.to_pandas()
            list_of_dicts = df.to_dict('records')
            dir_data.extend(list_of_dicts)
        zip_files = glob.glob(f'{dir}/train*.zip')
        img_dict = {}
        for zip_file in zip_files:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                all_files = zip_ref.namelist()
                for i, img_files in enumerate(all_files):
                    img = zip_ref.read(img_files)
                    img = io.BytesIO(img)
                    img_dict[img_files] = img
        print ('images', len(img_dict), 'examples', len(dir_data))
        for exp in dir_data:
            #uid = f"{dir_name}_{exp['id']}"
            uid_count[dir_name] += 1
            final_uid = f"{dir_name}_{uid_count[dir_name]}"
            uids[final_uid] = 1
            img_list = [img_dict[im['path']] for im in exp['images']]
            img_list = pickle.dumps(img_list)
            sink.write({
                    "__key__": final_uid,
                    "input_image": img_list,
                    "conversations": str(list(exp['conversation'])),
                })
    print ('final uids', len(final_uid))
    # table = pq.read_table('/data1/LLAVA_finetune_images/TIGER-Lab/Mantis-Instruct/chartqa/train-00000-of-00001.parquet')
    # df = table.to_pandas()
    # print (df.info())
    # print (df.head())
    # list_of_dicts = df.to_dict('records')
    # print (len(list_of_dicts))
    # with zipfile.ZipFile('/data1/LLAVA_finetune_images/TIGER-Lab/Mantis-Instruct/chartqa/train_images.zip', 'r') as zip_ref:
    #     all_files = zip_ref.namelist()
    #     # print (len(all_files))
    #     # print (all_files[:10])
    #     img_dict = {}
    #     for i, img_files in enumerate(all_files):
    #         img = zip_ref.read(img_files)
    #         img = io.BytesIO(img)
    #         img_dict[img_files] = img
    # print (len(img_dict))
    
    # for exp in tqdm(total_data):
    #     sink.write(exp)
        # img_list = [img_dict[im['path']] for im in exp['images']]
        # img_list = pickle.dumps(img_list)
        # sink.write({
        #         "__key__": exp["id"],
        #         "input_image": img_list,
        #         "conversations": str(exp['conversation']),
        #     })

import tarfile
import mmap
import struct
import collections
TarHeader = collections.namedtuple(
    "TarHeader",
    [
        "name",
        "mode",
        "uid",
        "gid",
        "size",
        "mtime",
        "chksum",
        "typeflag",
        "linkname",
        "magic",
        "version",
        "uname",
        "gname",
        "devmajor",
        "devminor",
        "prefix",
    ],
)
def parse_tar_header(header_bytes):
    header = struct.unpack("!100s8s8s8s12s12s8s1s100s6s2s32s32s8s8s155s", header_bytes)
    return TarHeader(*header)


def next_header(offset, header):
    block_size = 512
    size = header.size.decode("utf-8").strip("\x00")
    if size == "":
        return -1
    size = int(size, 8)
    # compute the file size rounded up to the next block size if it is a partial block
    padded_file_size = (size + block_size - 1) // block_size * block_size
    return offset + block_size + padded_file_size

def check_tar_files():
    # 
    stream = open("/data1/LLAVA_finetune_images/TIGER-Lab/mantis-instruct-000027.tar", "rb")
    mmapped_file = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
    print ('mmap', len(mmapped_file))
    by_name = {}
    by_index = []
    offset = 0
    while offset >= 0 and offset < len(mmapped_file):
        header = parse_tar_header(mmapped_file[offset : offset + 500])
        name = header.name.decode("utf-8").strip("\x00")
        typeflag = header.typeflag.decode("utf-8").strip("\x00")
        if name != "" and name != "././@PaxHeader" and typeflag in ["0", ""]:
            try:
                size = int(header.size.decode("utf-8")[:-1], 8)
            except ValueError as exn:
                print(header)
                raise exn
            by_name[name] = offset
            by_index.append((name, offset, size))
        else:
            print ('name', name)
            exit()
        offset = next_header(offset, header)
    print ('by name', len(by_name))
    exit()
    tar_file = tarfile.open("/data1/LLAVA_finetune_images/TIGER-Lab/mantis-instruct-000027.tar", "r")
    all_names = []
    for member in tqdm(tar_file):
        if member.isfile():
            all_names.append(member.name)
    print (len(all_names))
    for n in all_names:
        if n.startswith('dvqa_dvqa_reasoning_question_114432002_114432003_114432004_114432005_114432006_114432007_114432008_1'):
            print (n)
        if '.' not in n:
            print (n)
    print (all_names[:10])

if __name__ == '__main__':
    # prepare_webdataset_pretrain()
    # prepare_webdataset()
    # prepare_llava_instruct_new()
    prepare_llava_pretrain_new()
    #convert_mantis_webdataset()
    #check_tar_files()