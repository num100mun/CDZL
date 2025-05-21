from gen_utils import shuffle_generation
import time
def run(args,
        img_name,
        img_pil_list,
        lm_model,
        clip,
        yolos,
        token_mask,
        logger,
        all_results,
        all_clip_scores,
        init_sentence = ""):

    image_instance = img_pil_list
    start_time = time.time()
    gen_texts, clip_scores = shuffle_generation(args,
                                                img_name,
                                                lm_model,
                                                clip,
                                                yolos,
                                                image_instance,
                                                token_mask,
                                                args.prompt, logger,
                                                run_type=args.run_type,
                                                batch_size=args.batch_size,
                                                max_len=args.sentence_len,
                                                top_k=args.candidate_k,
                                                alpha=args.alpha,
                                                beta=args.beta,
                                                delata=args.delata,
                                                temperature=args.lm_temperature,
                                                max_iters=args.num_iterations,
                                                ctl_signal=args.control_type,
                                                init_method=args.init_method,
                                                using_yolos=args.using_yolos,
                                                init_sentence=init_sentence)

    logger.info("Finished in %.3fs" % (time.time() - start_time))

    #打印clipscore最好两轮的数据
    final_caption = gen_texts[-2]
    best_caption = gen_texts[-1]
    for i in range(args.batch_size):
        logger.info(f"The {i + 1}-th image: {img_name[i]}")
        logger.info(f"final caption: {final_caption[i]}")
        logger.info(f"best caption: {best_caption[i]}")

    #返回所有伦生成的数据
    for iter_id, gen_text_list in enumerate(gen_texts):
        for jj in range(len(gen_text_list)):
            image_id = img_name[jj].split(".")[0][-6:]
            if all_results[iter_id] == None:
                all_results[iter_id] = {image_id: gen_text_list[jj]}
                all_clip_scores[iter_id] = {image_id: clip_scores[jj]}

            else:
                all_results[iter_id][image_id] = gen_text_list[jj]
                all_clip_scores[iter_id][image_id] = clip_scores[jj]
    return all_results,  clip_scores




