from supersied import get_supersied_eval
from unsupersied import get_unsupersied_eval
import os
import json
from clip.clip import CLIP
from supersied import cocoeval
from arg import get_args
import re
#打印所有评价指标
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
def read_eval(path):
    read_parh = os.path.join(path, 'supersied_eval.json')
    json_data = json.load(open(read_parh, 'r'))
    for key, value in json_data.items():
        if key != 'clip_score':
            print(f'{key} : {round(value,4)}')
def supersied_eval(args, path = None):
    get_supersied_eval(path, clip=None)

def unsupersied_eval(args, path = None):
    get_unsupersied_eval(path, args.coco_train2014df)

def get_init_value(path):
    json_data = json.load(open(os.path.join(path, 'ini_process.json'), 'r'))
    r, a, _ = cocoeval(json_data)
    json.dump(r, open(os.path.join(path, 'supersied_eval.json'), 'w'))




mh_bert = r'C:\Users\zx\Desktop\Cdzl-new\results\caption_shuffle_random_len12_model\xlm-roberta-base_topk200_gennum5000_2025.05.07-13.19.08-mh_robert\result-4999\sample_0'
ob_sentence = r'C:\Users\zx\Desktop\Cdzl-new\results\caption_shuffle_random_len12_model\xlm-roberta-base_topk200_gennum5000_2025.05.08-04.46.19-ob_sentence\result-4999\sample_0'
cdzl = r'C:\Users\zx\Desktop\Cdzl-new\results\caption_shuffle_random_len12_model\xlm-roberta-base_topk200_gennum5000_2025.05.10-00.18.21-cdzl\result-4999\sample_0'
args = get_args(config_path="../setting.yaml")
read_eval(cdzl)


