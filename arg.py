import argparse
import yaml


def get_args(config_path=None):
    parser = argparse.ArgumentParser()

    # 在这里添加所有的argparse参数
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1, help="support batch_size>1 currently.")
    parser.add_argument("--device", type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument("--gen_num", type=int, default=1)
    parser.add_argument("--save_frq", type=int, default=10)
    parser.add_argument("--mask_num", type=int, default=1)
    parser.add_argument("--using_yolos", type=bool, default=True)
    parser.add_argument('--skip_num', type=int, default=0)
    parser.add_argument('--init_method', default='random', choices=['random', 'sentence'])
    parser.add_argument('--object_detector', type=bool, default=True)
    parser.add_argument('--run_type', default='caption', nargs='?', choices=['caption', 'controllable', 'init'])
    parser.add_argument('--prompt', default='Image of a ', type=str)
    parser.add_argument('--order', default='shuffle', nargs='?', choices=['sequential', 'shuffle', 'span', 'random'],
                        help="Generation order of text")
    parser.add_argument('--control_type', default='sentiment', nargs='?', choices=["sentiment", "pos"],
                        help="which controllable task to conduct")
    parser.add_argument('--pos_type', type=list,
                        default=[['DET'], ['ADJ', 'NOUN'], ['NOUN'], ['VERB'], ['VERB'], ['ADV'], ['ADP'],
                                 ['DET', 'NOUN'], ['NOUN'], ['NOUN', '.'], ['.', 'NOUN'], ['.', 'NOUN']],
                        help="predefined part-of-speech templete")
    parser.add_argument('--sentiment_type', default="positive", nargs='?', choices=["positive", "negative"])
    parser.add_argument('--samples_num', default=1, type=int)
    parser.add_argument("--sentence_len", type=int, default=12)
    parser.add_argument("--candidate_k", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.02, help="weight for fluency")
    parser.add_argument("--beta", type=float, default=2.0, help="weight for image-matching degree")
    parser.add_argument("--gamma", type=float, default=5.0, help="weight for controllable degree")
    parser.add_argument("--delata", type=float, default=0.5, help="weight for smal object degree")
    parser.add_argument("--fz", type=float, default=1, help="weight reject")
    parser.add_argument("--lm_temperature", type=float, default=0.1)
    parser.add_argument("--num_iterations", type=int, default=10, help="predefined iterations for Gibbs Sampling")
    parser.add_argument("--lm_model", type=str, default='D:\\model_file\\bert-base-uncased',
                        help="Path to language model")  # bert,roberta
    parser.add_argument("--match_model", type=str, default='D:\\model_file\\clip-vit-base-patch32',
                        help="Path to Image-Text model")  # clip,align
    parser.add_argument("--detection_model", type=str, default='D:\\model_file\\yolos-small', )
    parser.add_argument("--caption_img_path", type=str, default='D:\\model_file',
                        help="file path of imagess for captioning")
    parser.add_argument("--caption_json_data", type=str,
                        default="D:\\model_file\\Cdzl-requement\\coco_karpathy_val.json")
    parser.add_argument("--senticap_json_data", type=str,
                        default="D:\\model_file\\Cdzl-requement\\senticap_dataset.json")
    parser.add_argument("--stop_words_path", type=str, default="D:\\model_file\\Cdzl-requement\\stop_words.txt",
                        help="Path to stop_words.txt")
    parser.add_argument("--vocab_words", type=str, default="D:\\model_file\\Cdzl-requement\\vocab_words.txt")
    parser.add_argument("--vocab_words_pkl", type=str, default="D:\\model_file\\Cdzl-requement\\vocab_words.pkl")
    parser.add_argument("--coco_train2014df", type=str, default="D:\\model_file\\Cdzl-requement\\coco-train2014-df.p")
    parser.add_argument("--init_caption", type=str, default='model\\result-4050-pro.json')
    parser.add_argument("--add_extra_stopwords", type=list, default=[], help="you can add some extra stop words")
    parser.add_argument("--llm_model", type=str, default='')
    parser.add_argument("--gsm_model", type=str, default='')
    parser.add_argument("--adm_model", type=str, default='')
    parser.add_argument("--yolos_threshold", type=float, default=0.8)
    parser.add_argument("--use_mergeanddup", type=bool, default=True)
    parser.add_argument("--use_duplicated", type=bool, default=True)
    parser.add_argument("--use_confdience", type=bool, default=True)
    parser.add_argument("--use_post", type=bool, default=True)
    parser.add_argument("--use_correlation", type=bool, default=True)

    # 如果传入了配置文件路径，则加载yaml文件
    if config_path:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # 将配置更新到argparse参数中
        args = parser.parse_args()
        for key, value in config.items():
            if hasattr(args, key):  # 通过args来判断是否有对应的属性
                setattr(args, key, value)

    else:
        args = parser.parse_args()

    return args
