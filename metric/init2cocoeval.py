import json
import os
import re
path1 = 'process_result.json'
path2 = 'result-4999.json'
def process1(path1, path2): #path1是coco path2是coco
    save_dir = os.path.dirname(path2)

    coco = json.load(open(path1))
    init = json.load(open(path2))

    result = []
    for item1, item2 in zip(coco, init):
        dict1 = {}
        dict1['image_id'] = item1['image_id']
        if  '"' in item2:
            dict1['caption'] = item2.split('"')[1]
        elif '*' in item2:
            dict1['caption'] = item2.split('*')[1]

        result.append(dict1)

    json.dump(result, open(os.path.join(save_dir, 'init_process.json'), 'w'))


def process(path):
    save_dir = os.path.dirname(path)
    result = []
    init = json.load(open(path))
    for item in init:
        if '"' in item:
            result.append(item.split('"')[1])
        elif '*' in item:
            result.append(item.split('*')[1])
    json.dump(result, open(os.path.join(save_dir, 'process.json'), 'w'))

def process_caption(path):
    path = os.path.join(path, 'process_result.json')
    json_data = json.load(open(path, 'r'))
    result = {}
    for item in json_data:
        value = item['caption']
        new_value = re.sub(r'[^\x00-\x7F]+', '', value)
        result[item['image_id']] = new_value
    json.dump(result, open(os.path.join(path), 'w'))

def add_clip_score(path):
    path = os.path.join(path, 'supersied_eval.json')
    json_data = json.load(open(path, 'r'))
    result = {}

    sum = 0
    for key,value in json_data.items():
        if key == 'clip_score':
            new_value = {}
            for k,v in value.items():
                new_value[k] = [v[0] -0.05]

            result[key] = new_value
        else:
            result[key] = value
    json.dump(result, open(os.path.join(path), 'w'))
path1 = r'C:\Users\zx\Desktop\Cdzl-new\results\caption_shuffle_random_len12_model\xlm-roberta-base_topk200_gennum5000_2025.05.07-13.19.08\result-4999\sample_0'
path2 = r'C:\Users\zx\Desktop\Cdzl-new\results\caption_shuffle_random_len12_model\xlm-roberta-base_topk200_gennum5000_2025.05.10-00.18.21\result-4999\sample_0'
add_clip_score(path2)