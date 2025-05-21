
import json
import glob
import os.path


def gets(s):
    score = s.split(':')[1][-5:]
    img = s.split(',')[1][-10:-4]
    sentence = s.split(':')[-1].strip().replace('</s>', '')
    # s1 = ''
    # for i in sentence:
    #     if (ord('a') <= ord(i) and ord('z') >= ord(i) ) or i == ' ' or i =='.':
    #         s1 = s1 + i
    return score, img, sentence
def log2file(save_path , log_path):
    file = open(log_path,'r' ,encoding='utf-8')
    cur_img = ''
    temp = []
    it_list = [{},{},{},{},{}]
    best_clip = {}
    result = []
    supersied_eval = {"clip_score":{}}
    num = 0
    for i in file.readlines():
        it = i.split(',')[0].split('r')[-1].strip()
        if it < '1' or it > '5':
            continue
        if len(temp) == 5:
            temp = sorted(temp, key=lambda x:x[0], reverse=True)
            best_clip[cur_img] = temp[0][1]
            supersied_eval["clip_score"][cur_img] =[float(temp[0][0])]
            temp = []
        num = num + 1
        score, img, sentence = gets(i)
        cur_img = img
        temp.append([score, sentence])
        it_list[ord(it) - ord('1')][img] =sentence
    for key, value in best_clip.items():
        result.append({'image_id': int(key.split('.')[0][-6:]), 'caption': value})
    for i in range(len(it_list)):
        json.dump(it_list[i], open(os.path.join(save_path, f'iter_{i}.json'), 'w'))
    json.dump(best_clip, open(os.path.join( save_path, 'best_clipscore.json') , 'w'))
    json.dump(result, open(os.path.join( save_path, 'pro.json') , 'w'))
    json.dump(supersied_eval, open(os.path.join(save_path, 'supersied_eval.json'), 'w'))

log2file(r'D:\Desktop\fsdownload\sentence-robert-it5',r"D:\Desktop\fsdownload\robert-caption-it5-fz0.5.log")
# sum = 1.30 * 0.01+	13.28* 0.01	+17.04* 0.01	+8.20* 0.01		+0.99	+0.85	+0.76	+0.83
# print(sum)