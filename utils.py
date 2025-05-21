import json

import numpy as np
import os
import colorlog
import random
import torch
from datashader.core import LinearAxis

from model.llm import init_generate_caption
from datetime import datetime
def create_logger(folder, filename):
    log_colors = {
        'DEBUG': 'blue',
        'INFO': 'white',
        'WARNING': 'green',
        'ERROR': 'red',
        'CRITICAL': 'yellow',
    }

    import logging
    logger = logging.getLogger('Cdzl')
    # %(filename)s$RESET:%(lineno)d
    # LOGFORMAT = "%(log_color)s%(asctime)s [%(log_color)s%(filename)s:%(lineno)d] | %(log_color)s%(message)s%(reset)s |"
    LOGFORMAT = ""
    LOG_LEVEL = logging.DEBUG
    logging.root.setLevel(LOG_LEVEL)
    stream = logging.StreamHandler()
    stream.setLevel(LOG_LEVEL)
    stream.setFormatter(colorlog.ColoredFormatter(LOGFORMAT, datefmt='%d %H:%M', log_colors=log_colors))

    # print to log file
    hdlr = logging.FileHandler(os.path.join(folder, filename))
    hdlr.setLevel(LOG_LEVEL)
    # hdlr.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
    hdlr.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(hdlr)
    logger.addHandler(stream)
    return logger
def get_curtime():
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y.%m.%d-%H.%M.%S")
    return formatted_datetime
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def get_init_text(args, run_type, clip, yolos, tokenizer, seed_text, max_len, batch_size, init_method, image_instance, image_embeds):
    """ Get initial sentence by padding seed_text with [mask] words to max_len """
    text = seed_text + tokenizer.mask_token * max_len
    ids = tokenizer.encode(text)
    batch = [ids] * batch_size
    return batch,None
def get_init_text_sentence(args, run_type, clip, yolos, tokenizer, seed_text, max_len, batch_size, init_method, image_instance, image_embeds, sentence):
    """ Get initial sentence by padding seed_text with [mask] words to max_len """
    text = seed_text + sentence
    ids = tokenizer.encode(text)
    batch = [ids] * batch_size
    return batch,None

def random_sample_list(input_list, n):
    result = []
    while input_list:
        if len(input_list) < n:
            n = len(input_list)
        sample = random.sample(input_list, n)
        result.extend(sample)
        input_list = [item for item in input_list if item not in sample]
    return result
def update_token_mask(tokenizer, token_mask, max_len, index):
    """ '.'(full stop) is only allowed in the last token position """
    for i in index:
        if i == max_len:
            token_mask[:, tokenizer.vocab['.']] = 1
        else:
            token_mask[:, tokenizer.vocab['.']] = 0
    return token_mask
def format_output(sample_num, FinalCaption, BestCaption):
    if sample_num == 1:
        return f"{FinalCaption[0]}", f"{BestCaption[0]}"
    elif sample_num ==2:
        return f"{FinalCaption[0]}\n{FinalCaption[1]}", f"{BestCaption[0]}\n{BestCaption[1]}"
    elif sample_num ==3:
        return f"{FinalCaption[0]}\n{FinalCaption[1]}\n{FinalCaption[2]}",\
            f"{BestCaption[0]}\n{BestCaption[1]}\n{BestCaption[2]}"
    elif sample_num ==4:
        return f"{FinalCaption[0]}\n{FinalCaption[1]}\n{FinalCaption[2]}\n{FinalCaption[3]}",\
            f"{BestCaption[0]}\n{BestCaption[1]}\n{BestCaption[2]}\n{BestCaption[3]}"
    else:
        return f"{FinalCaption[0]}\n{FinalCaption[1]}\n{FinalCaption[2]}\n{FinalCaption[3]}\n{FinalCaption[4]}",\
            f"{BestCaption[0]}\n{BestCaption[1]}\n{BestCaption[2]}\n{BestCaption[3]}\n{BestCaption[4]}"
def load_init_json(path):
    return json.load(open(path))
