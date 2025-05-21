import numpy as np
import torch
import torch.nn.functional as F
import random
from utils import get_init_text, update_token_mask, get_init_text_sentence
import time
from utils import random_sample_list
from sentiments_classifer import batch_texts_POS_Sentiments_analysis


def generate_step(out, gen_idx,  temperature=None, top_k=0, sample=False, return_list=True):
    """ Generate a word from out[gen_idx]

    args:
        - out (torch.Tensor): tensor of logits of size batch_size x seq_len x vocab_size
        - gen_idx (int): location for which to generate for
        - top_k (int): if >0, only sample from the top k most probable words
        - sample (Bool): if True, sample from full distribution. Overridden by top_k
    """
    logits = out[:, gen_idx]
    if temperature is not None:
        logits = logits / temperature
    if top_k > 0:
        kth_vals, kth_idx = logits.topk(top_k, dim=-1)
        dist = torch.distributions.categorical.Categorical(logits=kth_vals)
        idx = kth_idx.gather(dim=1, index=dist.sample().unsqueeze(-1)).squeeze(-1)
    elif sample:
        dist = torch.distributions.categorical.Categorical(logits=logits)
        idx = dist.sample().squeeze(-1)
    else:
        idx = torch.argmax(logits, dim=-1)
    return idx.tolist() if return_list else idx

def generate_caption_step(out, gen_idxs, mask, temperature=None, top_k=100):
    # out, gen_idx=seed_len + ii, mask=token_mask, top_k=top_k, temperature=temperature
    """ Generate a word from out[gen_idx]
    args:
        - out (torch.Tensor): tensor of logits of size (batch_size, seq_len, vocab_size)
        - gen_idx (int): location for which to generate for
        - mask (torch.Tensor): (1, vocab_size)
        - top_k (int): candidate k
    """
    top_k_probs = []
    top_k_ids = []
    for gen_idx in gen_idxs:
        logits = out[:, gen_idx]
        if temperature is not None:
            logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        probs *= (mask)
        top_k_prob, top_k_id = probs.topk(top_k, dim=-1)
        top_k_probs.append(top_k_prob)
        top_k_ids.append(top_k_id)

    return torch.stack(top_k_probs), torch.stack(top_k_ids)

def shuffle_generation(args,
                       img_name,
                       model,
                       clip,
                       yolos,
                       image_instance,
                       token_mask,
                       prompt,
                       logger,
                       run_type='caption',
                       max_len=15,
                       top_k=100,
                       temperature=None,
                       alpha=0.7,
                       beta=1,
                       delata=1,
                       gamma=5,
                       ctl_signal='positive',
                       max_iters=20,
                       batch_size=1,
                       verbose=True,
                       init_method = 'random',
                       using_yolos=True,
                       init_sentence=""):
    print(init_sentence)
    seed_len = len(prompt.split())+1
    image_embeds = clip.compute_image_representation_from_image_instance(image_instance)
    #
    # #是否尝试获取先验知识
    if not init_sentence:
        batch,sentence = get_init_text(args,
                                       run_type,
                                       clip,
                                       yolos,
                                       model.lm_tokenizer,
                                       prompt,
                                       max_len,
                                       batch_size,
                                       init_method,
                                       image_instance,
                                       image_embeds)
    else:
        batch,sentence = get_init_text_sentence(args,
                                       run_type,
                                       clip,
                                       yolos,
                                       model.lm_tokenizer,
                                       prompt,
                                       max_len,
                                       batch_size,
                                       init_method,
                                       image_instance,
                                       image_embeds,
                                       init_sentence)

    inp = torch.tensor(batch).to(image_embeds.device)
    clip_score_sequence = []
    best_clip_score_list = [0] * batch_size
    best_caption_list = ['None'] * batch_size
    random_lst = list(range(len(batch[0]) - seed_len))
    random.shuffle(random_lst)
    logger.info(f"Order_list:{random_lst}")
    gen_texts_list = []

    if args.using_yolos:
        yolos.set_image_instance(image_instance)
    # #每个句子的循环次数
    for iter_num in range(max_iters):
        #循环单词长度
        for i in range(0, max(random_lst), args.mask_num):
            end = i + args.mask_num if i + args.mask_num < max(random_lst) else max(random_lst)
            ii = torch.tensor([random_lst[j] for j in range(i,end)])
            token_mask = update_token_mask(model.lm_tokenizer, token_mask, max(random_lst), ii)
            inp_copy = inp.clone()
            for j in ii:
                inp[:,seed_len + j] = model.lm_tokenizer.mask_token_id
            inp_ = inp.clone().detach()
            out = model.lm_model(inp).logits
            gen_indexs = ii + seed_len
            probs, idxs = generate_caption_step(out, gen_indexs,mask=token_mask, top_k=top_k, temperature=temperature)
            topk_inp = inp_.unsqueeze(1).repeat(1,top_k,1)
            idxs_ = (idxs * token_mask[0][idxs]).long()
            for num, j in enumerate(ii):
                topk_inp[:,:,j + seed_len] = idxs_[num]
            topk_inp_batch = topk_inp.view(-1,topk_inp.shape[-1])
            topk_inp_batch = torch.cat((topk_inp_batch, inp_copy), dim=0)
            batch_text_list= model.lm_tokenizer.batch_decode(topk_inp_batch , skip_special_tokens=True)
            clip_score, clip_ref = clip.compute_image_text_similarity_via_raw_text(image_embeds, batch_text_list)
            bert_score = probs.mean(dim=0)
            bert_score = torch.cat((bert_score, torch.mean(bert_score,dim=1).unsqueeze(0)), dim=1)
            final_score = 0
            if using_yolos:
                yolos_score = yolos.compute_score(batch_text_list, ii)
                final_score = final_score + delata * yolos_score
            if run_type == 'controllable' and  ctl_signal == 'sentiment':
                sentiment_probs_batch, sentiment_scores_batch, pos_tags, wordnet_pos_tags = batch_texts_POS_Sentiments_analysis(batch_text_list, 1,
                                                                                                                                topk_inp.device,
                                                                                                                                sentiment_ctl=ctl_signal,
                                                                                                                                batch_size_image=batch_size)
                final_score = final_score + gamma * sentiment_probs_batch
            final_score = alpha * bert_score + beta * clip_score + final_score
            final_score[0][-1] = final_score[0][-1] * args.fz
            best_clip_id = final_score.argmax(dim=1).view(-1,1)
            current_clip_score = clip_ref.gather(1, best_clip_id).squeeze(-1)
            clip_score_sequence_batch = current_clip_score.tolist()
            if best_clip_id == args.candidate_k:
                inp = inp_copy
            else:
                for num,j in enumerate(ii):
                    inp[:,seed_len + j] = idxs_[num].gather(1, best_clip_id).squeeze(-1)


        if verbose and np.mod(iter_num + 1, 1) == 0:
            for_print_batch = model.lm_tokenizer.batch_decode(inp)
            cur_text_batch= model.lm_tokenizer.batch_decode(inp,skip_special_tokens=True)
            for jj in range(batch_size):
                if best_clip_score_list[jj] < clip_score_sequence_batch[jj]:
                    best_clip_score_list[jj] = clip_score_sequence_batch[jj]
                    best_caption_list[jj] = cur_text_batch[jj]
                logger.info(f"iter {iter_num + 1}, The {jj+1}-th image: {img_name[jj]},"
                            f"clip score {clip_score_sequence_batch[jj]:.3f}: "+ for_print_batch[jj])
        gen_texts_list.append(cur_text_batch)
        clip_score_sequence.append(clip_score_sequence_batch)
    gen_texts_list.append(best_caption_list)
    clip_score_sequence.append(best_clip_score_list)

    return gen_texts_list, clip_score_sequence

