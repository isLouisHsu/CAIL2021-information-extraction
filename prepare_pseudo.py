import os
import json
from argparse import ArgumentParser
from prepare_corpus import load_cail2018_corpus
from utils import LABEL_MEANING_MAP

def main(args):
    args.output_dir = os.path.join(args.output_dir, 
        f"pseudo-minlen{args.min_length}-maxlen{args.max_length}-seed{args.seed}")
    os.makedirs(args.output_dir, exist_ok=True)

    corpus = load_cail2018_corpus(args.raw_files)
    # 保留句子长度超过`min_length`的
    length_filter_func = lambda x: len(x) > args.min_length and len(x) < args.max_length
    corpus = list(filter(length_filter_func, corpus))

    f = open(os.path.join(args.output_dir, "xxcq_pseudo.json"), "w", encoding="utf-8")
    entities = [{"label": label, "span": []} for label in LABEL_MEANING_MAP.keys()]
    for i, sentence in enumerate(corpus):
        f.write(json.dumps({
            "id": f"pseudo{i:025d}",
            "context": sentence,
            "entities": entities,
        }, ensure_ascii=False) + "\n")
    f.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="../cail_processed_data/")
    parser.add_argument("--min_length", type=int, default=20)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--raw_files", type=str, nargs="+", default=[
            "../cail_raw_data/2018/CAIL2018_ALL_DATA/final_all_data/exercise_contest/data_train.json",
            "../cail_raw_data/2018/CAIL2018_ALL_DATA/final_all_data/exercise_contest/data_valid.json",
            "../cail_raw_data/2018/CAIL2018_ALL_DATA/final_all_data/exercise_contest/data_test.json",
            "../cail_raw_data/2018/CAIL2018_ALL_DATA/final_all_data/first_stage/train.json",
            "../cail_raw_data/2018/CAIL2018_ALL_DATA/final_all_data/first_stage/test.json",
            "../cail_raw_data/2018/CAIL2018_ALL_DATA/final_all_data/restData/rest_data.json",
            "../cail_raw_data/2018/CAIL2018_ALL_DATA/final_all_data/final_test.json",
        ])
    parser.add_argument("--seed", default=42, type=int, help="Seed.")
    args = parser.parse_args()

    main(args)
