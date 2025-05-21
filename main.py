import json

import utils
from arg import get_args
from utils import set_seed, create_logger, get_curtime
import os
import time
from model.lm_model import lm_model
from model.clip import CLIP
from model.yolo import Yolo
from model.llm import  Llm
from model.adm import Adm
from model.gsm import Gsm
import model.llm as llm
import torch
from data import collate_img, Imgdata, save, save_init, ImgExampledata
from torch.utils.data import DataLoader
from tqdm import tqdm
from run import run

if __name__ == "__main__":
    #获取参数
    args = get_args(config_path="setting.yaml")
    set_seed(args.seed)

    #判断是否启用可控
    run_type = "caption" if args.run_type=="caption" else args.control_type
    if run_type=="sentiment":
        run_type = args.sentiment_type

    #启用Logger
    if os.path.exists("logger")== False:
        os.mkdir("logger")
    logger = create_logger( "logger",'{}_{}_len{}_topk{}_alpha{}_beta{}_gamma{}_lmtemp{}_{}.log'.format(run_type, args.order,args.sentence_len,args.candidate_k, args.alpha,args.beta,args.gamma,
                                                                                                        args.lm_temperature,time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())))
    #加载模型
    lm_model = ""  if (not args.lm_model) else  lm_model(args)
    logger.info("load {} success!".format(args.lm_model))
    clip = "" if (not args.match_model) else CLIP(args)
    logger.info("load {} success!".format(args.match_model))
    yolos = "" if (not args.detection_model) else Yolo(args, threshold=args.yolos_threshold)
    logger.info('load {} success!'.format(args.detection_model))
    llm_model = "" if (not args.llm_model) else Llm(args)
    logger.info("load {} success!".format(args.lm_model))
    adm_model = "" if (not args.adm_model) else Adm(args)
    logger.info("load {} success!".format(args.lm_model))
    gsm_model = "" if (not args.gsm_model) else Gsm(args)
    logger.info("load {} success!".format(args.lm_model))

    #加载停用词
    with open(args.stop_words_path, 'r', encoding='utf-8') as stop_words_file:
        stop_words = stop_words_file.readlines()
        stop_words_ = [stop_word.rstrip('\n') for stop_word in stop_words]
        stop_words_ += args.add_extra_stopwords
        stop_ids = lm_model.lm_tokenizer.convert_tokens_to_ids(stop_words_)
        token_mask = torch.ones((1, lm_model.lm_tokenizer.vocab_size))
        for stop_id in stop_ids:
            token_mask[0, stop_id] = 0
        token_mask = token_mask.to(args.device)

    #加载img数据
    img_data = Imgdata(args)
    # img_data = ImgExampledata(args)
    #获得img属性的迭代器
    train_loader = DataLoader(img_data, batch_size=args.batch_size, collate_fn=collate_img, shuffle=False, drop_last=True)

    #开始运行
    for sample_id in range(args.samples_num):
        all_results = [None] * (args.num_iterations+1)
        all_clip_scores = [None] * (args.num_iterations+1)
        train_loader = tqdm(train_loader)
        result = []
        init_sentence = []

        logger.info(f"Sample {sample_id + 1}: ")

        if args.run_type == 'cdzl':
            init_sentence = utils.load_init_json(args.init_caption)

        #遍历数据集
        for batch_idx, (img_batch_pil_list, name_batch_list) in enumerate(train_loader):
            logger.info(f"The {batch_idx+1}-th batch:")

            #开始记录时间
            start_time = time.time()

            with torch.no_grad():
                #OB-Sentence模型     run_type = init，  或者CDZL模型没有INIT也会启动OB-Sentence
                if args.run_type=="init" or (args.run_type=="cdzl" and args.init_caption==""):
                    ids, sentence = llm.init_generate_caption(args,
                                                            llm_model,
                                                            args.control_type,
                                                            args.vocab_words,
                                                            args.vocab_words_pkl,
                                                            clip,
                                                            yolos,
                                                            adm_model,
                                                            img_batch_pil_list,
                                                            None,
                                                            args.sentence_len,
                                                            args.prompt,
                                                            lm_model.lm_tokenizer,
                                                            args.use_correlation,
                                                            args.use_mergeanddup,
                                                            args.use_confdience,
                                                            args.use_post)
                    for i in sentence:
                        result.append(i)
                #CDZL模型
                if args.run_type == 'cdzl':
                    sentence = init_sentence[batch_idx]
                    all_results, clip_scores  = run(args,
                                                    name_batch_list,
                                                    img_batch_pil_list,
                                                    lm_model,
                                                    clip,
                                                    yolos,
                                                    token_mask,
                                                    logger,
                                                    all_results,
                                                    all_clip_scores,
                                                    sentence)

                #MH-Bert模型
                if args.run_type ==  'caption':
                    all_results, clip_scores  = run(args,
                                                    name_batch_list,
                                                    img_batch_pil_list,
                                                    lm_model, clip,yolos,
                                                    token_mask,
                                                    logger,
                                                    all_results,
                                                    all_clip_scores)
            logger.info(time.time() - start_time)

            #保存
            if (batch_idx*args.batch_size%args.save_frq == 0 or batch_idx == args.gen_num - 1 ):

                #获取保存路径
                save_dir = "results/caption_%s_%s_len%d_%s_topk%d_gennum%d_%s/result-%d/sample_%d" \
                           % ( args.order, args.init_method,
                              args.sentence_len, args.lm_model.split('\\')[-1],
                              args.candidate_k, args.gen_num, get_curtime(),
                               batch_idx*args.batch_size, sample_id)

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                #保存参数
                args_dict = vars(args)
                with open(os.path.join(save_dir, 'args.json'), 'w', encoding='utf-8') as f:
                    json.dump(args_dict, f, ensure_ascii=False, indent=4)

                #保存结果
                if args.run_type == 'init':
                    with open(f'result-{batch_idx}.json', 'w') as f:
                        json.dump(result, f)
                else:
                    save(save_dir, all_results, all_clip_scores)

                # #评价模型
                # from metric import eval
                # eval.get_eval(args, save_dir)
                # eval.read_eval(save_dir)


            #生成数量
            if batch_idx == args.gen_num:
                break

