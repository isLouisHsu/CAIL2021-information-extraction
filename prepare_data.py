import os
import copy
import json
import utils
import random
import logging
from argparse import ArgumentParser
from sklearn.model_selection import KFold

logging.getLogger().setLevel(logging.INFO)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_files", type=str, nargs="+", 
        default=["./data/信息抽取_第一阶段/xxcq_small.json"],  
        help="Data files.")
    parser.add_argument("--context_window", default=0, type=int, 
        help="Size of context window.")
    parser.add_argument("--n_splits", default=5, type=int)
    parser.add_argument("--output_dir", type=str, default="./data/")
    parser.add_argument("--seed", default=42, type=int, 
        help="Seed.")
    args = parser.parse_args()

    # prepare
    utils.seed_everything(args.seed)
    args.output_dir = os.path.join(args.output_dir,
        f"ner-ctx{args.context_window}-{args.n_splits}fold-seed{args.seed}")
    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f"Saving to {args.output_dir}")

    # load raw
    raw_samples = []
    for data_file in args.data_files:
        raw_samples.extend(utils.load_raw(data_file))
        # TODO: [-STARTDOC-]
    num_samples = len(raw_samples)
    logging.info(f"Number of raw samples: {num_samples}")

    # context window
    raw_samples = utils.add_context(raw_samples, args.context_window)

    # # split
    # random.shuffle(raw_samples)
    # num_train = int(num_samples * args.train_split_ratio)
    # num_dev = num_samples - num_train
    # train_samples = raw_samples[: num_train]
    # dev_samples = raw_samples[num_train:]
    # logging.info(f"Number of training data: {num_train}, number of dev data: {num_dev}")

    # # save samples
    # utils.save_samples(os.path.join(args.output_dir, "train.json"), train_samples)
    # utils.save_samples(os.path.join(args.output_dir, "dev.json"), dev_samples)

    # k-fold split
    kf = KFold(n_splits=args.n_splits, shuffle=True)
    for k, (train_index, dev_index) in enumerate(kf.split(raw_samples)):
        train_samples = [raw_samples[idx] for idx in train_index]
        dev_samples = [raw_samples[idx] for idx in dev_index]
        utils.save_samples(os.path.join(args.output_dir, f"train_{k}.json"), train_samples)
        utils.save_samples(os.path.join(args.output_dir, f"dev_{k}.json"), dev_samples)
