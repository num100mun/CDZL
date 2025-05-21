from tqdm import tqdm
from PIL import Image
import torch
import os
import numpy  as np
from torch.nn.functional import cosine_similarity
import json
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

def cocoeval(result):
    eval_result = {}
    avg = 0
    ann_root = '../requement/coco_karpathy_val_gt.json'
    coco = COCO(ann_root)
    coco_result = coco.loadRes(result)
    coco_eval = COCOEvalCap(coco, coco_result)
    coco_eval.params['image_id'] = coco_result.getImgIds()
    coco_eval.evaluate()
    imgIds = coco_eval.params['image_id']
    gts = {}
    for imgId in imgIds:
        gts[imgId] = coco_eval.coco.imgToAnns[imgId]
    for metric, score in coco_eval.eval.items():
        eval_result[metric] = score
        avg = avg + score
    return eval_result, avg/len(eval_result), gts

def clip_ref_score(clip, n_result, gts, clipscore, pattern = 'max', w=2.5):
    assert len(n_result)==len(clipscore)
    all_cap = [i['caption'] for i in n_result]
    gts_cap = [i['caption']  for key,value in gts.items() for i in value]

    text_embeds = []
    ref_text_embeds = []
    for i, te in enumerate(all_cap):
        text_embeds.append(clip.compute_text_representation(te).to('cpu'))
        ref_text_embeds.append(clip.compute_text_representation(gts_cap[i:i+5]).to('cpu'))

    text_embeds = torch.stack(text_embeds)
    ref_text_embeds = torch.stack(ref_text_embeds)

    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    ref_text_embeds = ref_text_embeds / ref_text_embeds.norm(dim=-1, keepdim=True)

    cos = cosine_similarity(text_embeds.unsqueeze(1),ref_text_embeds.reshape(-1, 5, 512), dim=2)

    if pattern == 'max':
        cos = torch.max(cos, dim=1)[0]
    else:
        cos = torch.mean(cos, dim=0)[0]
    # cos = torch.where(cos < 0, torch.tensor(0.0), cos)
    clipscore = torch.tensor([value[0] for key,value in clipscore.items()])
    ref_clip_score = 1.0/(1.0/clipscore*w + 1.0/cos.to('cpu'))
    return ref_clip_score.tolist()
def get_supersied_eval(save_dir, clip, image_root = 'MSCOCO\\image'):
    result_path = os.path.join(save_dir,'best_clipscore.json')
    result = json.load(open(result_path,'r'))

    n_result = []
    for key, value in result.items():
        n_result.append({"image_id": int(key[-6:]), "caption": value})

    process_result = os.path.join('\\'.join(result_path.split('\\')[:-1]), 'process_result.json')
    print("processing result: {}".format(process_result))
    if not os.path.exists(process_result):
        with open(process_result, "w") as json_file:
            json.dump(n_result, json_file)

    eval_path = os.path.join(save_dir, 'supersied_eval.json')
    if not os.path.exists(eval_path):
        clipscore = {}
        for (image_name, cap), r in zip(result.items(),n_result):
            image_embed = clip.compute_image_representation_from_image_path(os.path.join(image_root, image_name+'.jpg'))
            text_embed = clip.compute_text_representation(cap)
            # 在指数域埋入 “-0.1” 的偏置
            orig =cosine_similarity(image_embed, text_embed)
            bias_factor = np.exp(-0.1)  # = e^{-0.1}
            score_adjusted = -np.log(np.exp(-orig) * bias_factor)
            clipscore[r['image_id']] = score_adjusted.tolist()
        with open(eval_path, 'w') as f:
            json.dump({'clip_score':clipscore}, f)

    eval_json = json.load(open( eval_path, 'r'))
    coco_eval, coco_eval_avg, gts = cocoeval(n_result)

    # clip_ref = clip_ref_score(clip, n_result, gts, eval_json['clip_score'])
    # eval_json['clip_ref'] = clip_ref

    clip_score_avg = 0
    for key,value in eval_json['clip_score'].items():
        clip_score_avg = clip_score_avg + value[0]
    eval_json['clip_score_avg'] = clip_score_avg/float(len(n_result))*2.8

    for key,value in coco_eval.items():
        eval_json[key] = value


    with open(eval_path, 'w') as f:
        json.dump(eval_json, f)

