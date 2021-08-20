import os
import copy
import json
import utils
import random
import logging
from argparse import ArgumentParser

logging.getLogger().setLevel(logging.INFO)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_files", type=str, nargs="+", 
        default=["./data/信息抽取_第一阶段/xxcq_small.json"],  
        help="Data files.")
    parser.add_argument("--context_window", default=510, type=int, 
        help="Size of context window.")
    parser.add_argument("--train_split_ratio", default=0.8, type=float, 
        help="Size of training data.")
    parser.add_argument("--output_dir", type=str, default="./data/")
    parser.add_argument("--seed", default=42, type=int, 
        help="Seed.")
    args = parser.parse_args()

    # prepare
    utils.seed_everything(args.seed)
    args.output_dir = os.path.join(args.output_dir, 
        f"ner-ctx{args.context_window}-train{args.train_split_ratio}-seed{args.seed}")
    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f"Saving to {args.output_dir}")
    # load raw
    raw_samples = []
    for data_file in args.data_files:
        raw_samples.extend(utils.load_raw(data_file))
    num_samples = len(raw_samples)
    logging.info(f"Number of raw samples: {num_samples}")

    # context window
    if args.context_window > 0:

        for i in range(len(raw_samples)):
            if i == 0: continue
            text = raw_samples[i]["text"]
            add_left = (args.context_window-len(text)) // 2
            add_right = (args.context_window-len(text)) - add_left
            sent_start, sent_end = raw_samples[i]["sent_start"], raw_samples[i]["sent_end"]

            # add left context
            j = i - 1
            while j >= 0 and add_left > 0:
                context_to_add = raw_samples[j]["text"][-add_left:]
                text = context_to_add + text
                add_left -= len(context_to_add)
                sent_start += len(context_to_add)
                sent_end += len(context_to_add)
                j -= 1

            # add right context
            j = i + 1
            while j < len(raw_samples) and add_right > 0:
                context_to_add = raw_samples[j]["text"][:add_right]
                text = text + context_to_add
                add_right -= len(context_to_add)
                j += 1
            
            # adjust entities
            entities = []
            for label, start, end, span_text in raw_samples[i]["entities"]:
                start += sent_start; end += sent_start
                assert text[start: end] == span_text
                entities.append((label, start, end, span_text))
            
            raw_samples[i]["text"] = text
            raw_samples[i]["sent_start"] = sent_start
            raw_samples[i]["sent_end"] = sent_end
            raw_samples[i]["entities"] = entities

    # split
    random.shuffle(raw_samples)
    num_train = int(num_samples * args.train_split_ratio)
    num_dev = num_samples - num_train
    train_samples = raw_samples[: num_train]
    dev_samples = raw_samples[num_train:]
    logging.info(f"Number of training data: {num_train}, number of dev data: {num_dev}")

    # save
    def save(filename, samples):
        with open(filename, "w") as f:
            for sample in samples:
                sample = json.dumps(sample, ensure_ascii=False) + "\n"
                f.write(sample)
    save(os.path.join(args.output_dir, "train.json"), train_samples)
    save(os.path.join(args.output_dir, "dev.json"), dev_samples)
