import glob

from torch.utils.data import Dataset
import json
import os
from PIL import Image
class Imgdata(Dataset):
    def __init__(self,args):
        self.dir_path = args.caption_img_path
        self.run_type = args.run_type
        self.con_type = args.control_type
        if self.run_type == 'caption' or self.run_type == 'init' or self.run_type == 'cdzl':
            self.ann = json.load(open(args.caption_json_data, 'r'))
        if self.run_type == 'controllable':
            self.ann = json.load(open(args.senticap_json_data, 'r'))['images']

    def __getitem__(self, idx):
        if self.run_type == 'caption' or self.run_type == 'init' or self.run_type == 'cdzl':
            img_name = self.ann[idx]['image']
            img_item_path = os.path.join(self.dir_path, 'val2014', img_name)
        if self.run_type == 'controllable':
            img_name = self.ann[idx]['filename']
            img_item_path = os.path.join(self.dir_path, 'val2014', img_name)
        img = Image.open(img_item_path).convert("RGB")
        return img, img_name

    def __len__(self):
        return len(self.ann)

class ImgExampledata(Dataset):
    def __init__(self,args):
        self.all_path = glob.glob(os.path.join(args.caption_img_path, '*.jpg'))
    def __getitem__(self, idx):
        img_path = self.all_path[idx]
        img = Image.open(img_path).convert("RGB")
        return img, img_path
    def __len__(self):
        return len(self.all_path)
def collate_img(batch_data):
    img_path_batch_list = list()
    name_batch_list = list()
    for unit in batch_data:
        img_path_batch_list.append(unit[0])
        name_batch_list.append(unit[1])
    return img_path_batch_list, name_batch_list
def save_init(save_dir, result, art=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_dir, 'w') as _json:
        json.dump(result, _json)
def save(save_dir, all_results, all_clip_scores, art=None):
    # for iter_id in range(len(all_results)):
    #     if iter_id != len(all_results) - 1:
    #         cur_json_file = os.path.join(save_dir, f"iter_{iter_id}.json")
    #         with art.new_file(cur_json_file, 'w') as _json:
    #             json.dump(all_results[iter_id], _json)
    #     else:
    #         cur_json_file = os.path.join(save_dir, f"best_clipscore.json")
    #         cur_json_file_clips = os.path.join(save_dir, f"supersied_eval.json")
    #         with art.new_file(cur_json_file, 'w') as _json:
    #             json.dump(all_results[iter_id], _json)
    #         with art.new_file(cur_json_file_clips, 'w') as _json_clipscore:
    #             clipscore_result = {"clip_score": all_clip_scores[iter_id]}
    #             json.dump(clipscore_result, _json_clipscore)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for iter_id in range(len(all_results)):
        if iter_id != len(all_results) - 1:
            cur_json_file = os.path.join(save_dir, f"iter_{iter_id}.json")
            with open(cur_json_file, 'w') as _json:
                json.dump(all_results[iter_id], _json)
        else:
            cur_json_file = os.path.join(save_dir, f"best_clipscore.json")
            cur_json_file_clips = os.path.join(save_dir, f"supersied_eval.json")
            with open(cur_json_file, 'w') as _json:
                json.dump(all_results[iter_id], _json)
            with open(cur_json_file_clips, 'w') as _json_clipscore:
                clipscore_result = {"clip_score": all_clip_scores[iter_id]}
                json.dump(clipscore_result, _json_clipscore)

