import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import  pandas as pd
import string
import json
from torch.utils.data import Dataset,DataLoader
import argparse
import os
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from transformers import AutoModelForMaskedLM,AutoTokenizer
import torch
import numpy as np
from torch.nn.functional import cosine_similarity
stop_words = set(stopwords.words('english') + list(string.punctuation))
def get_word_sentiment(word, tag):
    n = ['NN', 'NNP', 'NNPS', 'NNS', 'UH']
    v = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    a = ['JJ', 'JJR', 'JJS']
    r = ['RB', 'RBR', 'RBS', 'RP', 'WRB']

    wn_tag = None
    if tag.startswith('J'):
        wn_tag = wn.ADJ
    elif tag.startswith('N'):
        wn_tag = wn.NOUN
    elif tag.startswith('R'):
        wn_tag = wn.ADV
    elif tag.startswith('V'):
        wn_tag = wn.VERB

    if wn_tag is None:
        return None

    synsets = wn.synsets(word, wn_tag)
    if not synsets:
        return None

    # 假设选择第一个同义词集（您可以根据需要选择不同的同义词集）
    synset = synsets[0]
    swn_synset = swn.senti_synset(synset.name())

    return swn_synset
def sentence_sentiment(sentence):
    # 分词并去除停用词
    tokens = word_tokenize(sentence)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    pos_tags = pos_tag(filtered_tokens)

    sentiment_scores = []
    for word, tag in pos_tags:
        sentiment = get_word_sentiment(word, tag)
        if sentiment:
            sentiment_scores.append(sentiment)

    if not sentiment_scores:
        return 0  # 如果没有找到情感词汇，则默认为中性

    # 计算情感得分的总和或平均值
    total_score = sum([sentiment.pos_score() - sentiment.neg_score() for sentiment in sentiment_scores])
    average_score = total_score / len(sentiment_scores)

    li = 0
    if average_score >= 0:
        li = 1
    else:
        li = 0

    return average_score,li

class Imgdata(Dataset):
    def __init__(self, dir_path, run_type, con_type):
        self.dir_path = dir_path
        self.run_type = run_type
        if run_type == 'caption':
            self.ann = json.load(open("D:\\软件安装包\\Cdzl-requement\\coco_karpathy_val.json", 'r'))
        if run_type == 'controllable':
            self.ann = json.load(open("D:\\软件安装包\\Cdzl-requement\\senticap_dataset.json", 'r'))['images']

    def __getitem__(self, idx):
        if self.run_type == 'caption':
            img_name = self.ann[idx]['image']
            img_item_path = os.path.join(self.dir_path, img_name)
            img = Image.open(img_item_path).convert("RGB")
            return img, img_name
        if self.run_type == 'controllable':
            img_name = self.ann[idx]['filename']
            img_item_path = os.path.join(self.dir_path, 'val2014', img_name)
            sentiment = [i['sentiment'] for i in self.ann[idx]['sentences']]
            img = Image.open(img_item_path).convert("RGB")
            setence = [i['raw'] for i in self.ann[idx]['sentences']]
            return img, img_name, sentiment, setence

    def __len__(self):
        return len(self.ann)
def collate_img(batch_data):
    img_path_batch_list = list()
    name_batch_list = list()
    se = list()
    s = list()
    for unit in batch_data:
        img_path_batch_list.append(unit[0])
        name_batch_list.append(unit[1])
        se.append(unit[2])
        s.append(unit[3])
    return img_path_batch_list, name_batch_list, se, s

def get_dataloader():
    img_data = Imgdata('D:\\软件安装包', 'controllable', 'con_type')
    train_loader = DataLoader(img_data, batch_size=1, collate_fn=collate_img, shuffle=False,drop_last=True)
    bar = tqdm(train_loader)
    return bar
    # grand_true = []
    # sen_value = []
    # for i,j,k,w in bar:
    #     grand_true.append(k[0])
    #     sen_value.append([sentence_sentiment(i)[1] for i in w[0]])
    #
    # sum1 = 0
    # l = 0
    # for i,j in zip(grand_true,sen_value):
    #     for k,w in zip(i,j):
    #         if k == w:
    #             sum1 = sum1 + 1
    #         l += 1
    # print(sum1, sum1/l)


def cocoeval(result):
    eval_result = {}
    avg = 0
    ann_root = 'D:\\软件安装包\\Cdzl-requement\\senticap_dataset.json'
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

def wsim(text1, text2):
    sim = []
    for i,j in zip(text1,text2):
        synsets_word1 = wn.synsets(i)
        synsets_word2 = wn.synsets(j)
        max_similarity = 0.0
        for synset1 in synsets_word1:
            for synset2 in synsets_word2:
                similarity = synset1.wup_similarity(synset2)  # 使用Wu-Palmer Similarity作为相似度度量
                if similarity is not None and similarity > max_similarity:
                    max_similarity = similarity
        if max_similarity < 0:
            sim.append(0)
        else:
            sim.append(float("{:.{dp}f}".format(max_similarity, dp=3)))
    return sim
def bsim(text1, text2):
    assert len(text1) ==len(text2)
    model = AutoModelForMaskedLM.from_pretrained('D:\\软件安装包\\bert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained('D:\\软件安装包\\bert-base-uncased')
    sim = []
    for i,j in zip(text1, text2):
        inputs = tokenizer([i,j], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            word1_embedding = outputs.logits[0][1].unsqueeze(0)
            word2_embedding = outputs.logits[1][1].unsqueeze(0)
        similarity = cosine_similarity(word1_embedding, word2_embedding,dim=1)
        sim.append(float("{:.{dp}f}".format(similarity.item(), dp=3)))
    return sim
print(bsim(['apple'],['top']))
print(wsim(['apple'],['top']))