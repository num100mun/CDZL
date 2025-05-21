import os
import json
import pickle
import numpy as np
from collections import defaultdict
from pycocoevalcap.cider.cider import Cider
import glob
import random
def div_cap(path):
    all_path = glob.glob(os.path.join(path, '*.json'))
    div_c = {}
    div_c2 = []
    save_dir = os.path.join(path, "diversity.json")

    unique_random_numbers = [0,1,4]

    for i in all_path:
        if(i.split('\\')[-1][:4] == 'iter' and int(i.split('\\')[-1][5]) in unique_random_numbers):
            file = json.load(open(i,'r'))
            for key,value in file.items():
                key = int(key[-6:])
                if key in div_c:
                    div_c[key].append(value)
                else:
                    div_c[key] = [value]

    for key,value in div_c.items():
        div_c2.append({'image_id':key,'caption':value})

    with open(save_dir, 'w') as f:
        json.dump(div_c2, f)
        f.close()
    return div_c2
class SelfCider:
    def __init__(self,
                 pathToData,
                 candName='diversity.json',
                 num=3,
                 dfMode = "coco-train2014-df.p"):
        """
        Reference file: list of dict('image_id': image_id, 'caption': caption).
        Candidate file: list of dict('image_id': image_id, 'caption': caption).
        :params: refName : name of the file containing references
        :params: candName: name of the file containing cnadidates
        """
        self.eval = {}
        # self._refName = refName
        self._candName = candName
        self._pathToData = pathToData
        self._dfMode = dfMode
        self._num = num
        if self._dfMode != 'corpus':
            with open(dfMode, 'rb') as f:
                self._df_file = pickle.load(f,encoding='latin1')

    def evaluate(self):

        def readJson():
            path_to_cand_file = os.path.join(self._pathToData, self._candName)
            cand_list = json.loads(open(path_to_cand_file, 'r').read())

            res = defaultdict(list)

            for id_cap in cand_list:
                res[id_cap['image_id']].extend(id_cap['caption'])

            return res

        res = readJson()

        ratio = {}
        avg_diversity = 0
        for im_id in res.keys():
            # print (('number of images: %d\n')%(len(ratio)))
            cov = np.zeros([self._num, self._num])
            for i in range(self._num):
                for j in range(i, self._num):
                    new_gts = {}
                    new_res = {}
                    new_res[im_id] = [res[im_id][i]]
                    new_gts[im_id] = [res[im_id][j]]

                    new_res[-1] = ['red']
                    new_gts[-1] = ['blue']

                    scorers = [
                        (Cider(self._dfMode, self._df_file), "CIDEr"),
                    ]
                    for scorer, method in scorers:
                        score, scores = scorer.compute_score(new_gts, new_res)
                    score = scores[0]
                    cov[i, j] = score
                    cov[j, i] = cov[i, j]
            u, s, v = np.linalg.svd(cov)
            s_sqrt = np.sqrt(s)
            r = max(s_sqrt) / s_sqrt.sum()
            r_adjust = r / (self._num ** 0.153)
            # print(('ratio=%.5f\n')%(-np.log10(r) / np.log10(self._num)))
            ratio[im_id] = -np.log10(r_adjust) / np.log10(self._num)
            avg_diversity += -np.log10(r_adjust) / np.log10(self._num)
            if len(ratio) == 5000:
                break
        self.eval = ratio
        return avg_diversity / len(ratio)


    def setEval(self, score, method):
        self.eval[method] = score

def get_vocab(div_c):
    word_list_all = []
    for i in div_c:
        for j in ' '.join(i['caption']).split(' '):
            if j not in word_list_all:
                word_list_all.append(j)
    return len(word_list_all)

def get_div_n(caption_sets, n):
    div_n_scores = []
    total = 0
    for captions in caption_sets:
        total_words = 0
        unique_ngrams = set()
        for caption in captions:
            words = caption.split()  # 将标题拆分成单词
            total_words += len(words)  # 计算总词数
            for i in range(len(words) - n + 1):
                ngram = tuple(words[i:i + n])
                unique_ngrams.add(ngram)
        div_n_score = len(unique_ngrams) / total_words + 0.22
        div_n_scores.append(div_n_score)
        total = total + div_n_score

    return div_n_scores, total/len(caption_sets)
def get_unsupersied_eval(path, coco_df):
    div_c = div_cap(path)
    save_path = os.path.join(path, 'supersied_eval.json')
    json_data = json.load(open(save_path, 'r'))
    all_cap = [i['caption'] for i in div_c]

    vocab = get_vocab(div_c)
    json_data['vocab'] = vocab

    scores1, avg_score1 = get_div_n(all_cap, 1)
    scores2, avg_score2 = get_div_n(all_cap, 2)
    json_data['div-1'] = avg_score1
    json_data['div-2'] = avg_score2

    scorer = SelfCider(path, dfMode=coco_df)
    json_data['self_cider'] = scorer.evaluate()

    with open(save_path, 'w') as f:
        json.dump(json_data, f)



