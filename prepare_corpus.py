# -*- coding: utf-8 -*-
'''
Description: 
Version: 
Author: louishsu
Github: https://github.com/isLouisHsu
E-mail: is.louishsu@foxmail.com
Date: 2021-09-19 14:53:15
LastEditTime: 2021-09-20 16:21:17
LastEditors: louishsu
FilePath: \CAIL2021-information-extraction\prepare_corpus.py
'''
import os
import re
import sys
import json
import random
from tqdm import tqdm
from collections import Counter
from argparse import ArgumentParser

def _strip(document):
    document = document.split("指控", 1)
    if len(document) == 2:
        document = document[1]
        if document[0] in ["：", "，"]:
            document = document[1:]
    else:
        document = document[0]
    return document

def _process(document):
    document = re.sub(r"\s+", "", document)
    # document = document.translate({ord(f): ord(t) for f, t in zip(
    #     u',.!?[]()<>"\'', u'，。！？【】（）《》“‘')})
    return document

def _split_doc(document, max_length=256):
    sentences = re.split(r"[。；;]", document)
    sentences = list(filter(lambda x: len(x) > 0, sentences))
    for i, sentence in enumerate(sentences):
        start_idx = document.find(sentence)
        end_idx = start_idx + len(sentence)
        if end_idx == len(document):
            continue
        sign = document[end_idx]
        sentences[i] = sentence + sign
    if max_length is not None:
        sentences_new = []
        sentence_new = ""
        for sentence in sentences:
            if len(sentence_new) + len(sentence) > max_length:
                sentences_new.append(sentence_new)
                sentence_new = ""
            sentence_new += sentence
        if len(sentence_new) > 0:
            sentences_new.append(sentence_new)
        sentences = sentences_new
    return sentences

def load_cail2018_corpus(filepaths):
    corpus = []
    accusations = set()
    for filepath in filepaths:
        with open(filepath, "r", encoding="utf-8") as f:
            # while True:
            #     line = f.readline()
            #     if line == "": break
            lines = f.readlines()
            for line in tqdm(lines, desc=f"{filepath}", total=len(lines)):
                line = json.loads(line.strip())
                accusation = ",".join(line["meta"]["accusation"])
                accusations.add(accusation)
                # if re.search(r"(抢劫|盗窃)", accusation) is None:
                if re.search(r"盗窃", accusation) is None:
                    continue
                document = line["fact"].strip()
                document = _process(document)
                document = _strip(document)
                sentences = _split_doc(document)
                sentences = [sentence.rstrip("。") + "。" for sentence in sentences if len(sentence) > 0]
                corpus.extend(sentences)
    print(f"{sys._getframe().f_code.co_name} #{len(corpus)}")
    return corpus

# def load_cail2020_ydlj_corpus(filepaths):
#     corpus = []
#     for filepath in filepaths:
#         with open(filepath, "r", encoding="utf-8") as f:
#             lines = json.load(f)
#             for line in lines:
#                 for sentence in line["context"][0][1]:
#                     corpus.append(_process(sentence))
#     print(f"{sys._getframe().f_code.co_name} #{len(corpus)}")
#     return corpus

# def load_cail2021_aqbq_corpus(filepaths):
#     """ 案情标签 """
#     corpus = []
#     for filepath in filepaths:
#         with open(filepath, "r", encoding="utf-8") as f:
#             lines = json.load(f)
#             for line in lines:
#                 for sentence in line["content"]:
#                     corpus.append(_process(sentence))
#     print(f"{sys._getframe().f_code.co_name} #{len(corpus)}")
#     return corpus

def load_cail2021_aljs_candidate_corpus(dirname):
    """ 案类检索 """
    corpus = []
    subdirs = os.listdir(dirname)
    for subdir in tqdm(subdirs, desc="Loading...", total=len(subdirs)):
        if subdir.startswith("."): continue
        subdir = os.path.join(dirname, subdir)
        for filename in os.listdir(subdir):
            filename = os.path.join(subdir, filename)
            with open(filename, "r", encoding="utf-8") as f:
                line = json.load(f)
                # if re.search(r"(抢劫|盗窃)", line["ajName"]) is None:
                if re.search(r"盗窃", line["ajName"]) is None:
                    continue
                for key in ["ajjbqk", "cpfxgc", "pjjg", "qw"]:
                    document = line.get(key, None)
                    if document is None: continue
                    document = _process(document)
                    document = _strip(document)
                    sentences = _split_doc(document)
                    sentences = [sentence + "。" for sentence in sentences if len(sentence) > 0]
                    corpus.extend(sentences)
    print(f"{sys._getframe().f_code.co_name} #{len(corpus)}")
    return corpus

def load_cail2021_ydlj_corpus(filepaths):
    """ 阅读理解 """
    corpus = []
    for filepath in filepaths:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = json.load(f)["data"]
            for line in lines:
                # if re.search(r"(抢劫|盗窃)", line["paragraphs"][0]["casename"]) is None:
                if re.search(r"盗窃", line["paragraphs"][0]["casename"]) is None:
                    continue
                document = line["paragraphs"][0]["context"]
                document = _process(document)
                document = _strip(document)
                sentences = _split_doc(document)
                sentences = [sentence + "。" for sentence in sentences if len(sentence) > 0]
                corpus.extend(sentences)
    print(f"{sys._getframe().f_code.co_name} #{len(corpus)}")
    return corpus

def load_cail2021_xxcq_corpus(filepaths):
    """ 信息抽取 """
    corpus = []
    for filepath in filepaths:
        with open(filepath, "r", encoding="utf-8") as f:
            while True:
                line = f.readline()
                if line == "": break
                line = json.loads(line.strip())
                document = line["context"].strip()
                document = _process(document)
                document = _strip(document)
                sentences = _split_doc(document)
                sentences = [sentence + "。" for sentence in sentences if len(sentence) > 0]
                corpus.extend(sentences)
    print(f"{sys._getframe().f_code.co_name} #{len(corpus)}")
    return corpus

def main(args):
    args.output_dir = os.path.join(args.output_dir, f"mlm-minlen{args.min_length}-maxlen{args.max_length}-seed{args.seed}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    corpus = []
    corpus.extend(
        load_cail2018_corpus([
            "../cail_raw_data/2018/CAIL2018_ALL_DATA/final_all_data/exercise_contest/data_train.json",
            "../cail_raw_data/2018/CAIL2018_ALL_DATA/final_all_data/exercise_contest/data_valid.json",
            "../cail_raw_data/2018/CAIL2018_ALL_DATA/final_all_data/exercise_contest/data_test.json",
            "../cail_raw_data/2018/CAIL2018_ALL_DATA/final_all_data/first_stage/train.json",
            "../cail_raw_data/2018/CAIL2018_ALL_DATA/final_all_data/first_stage/test.json",
            "../cail_raw_data/2018/CAIL2018_ALL_DATA/final_all_data/restData/rest_data.json",
            "../cail_raw_data/2018/CAIL2018_ALL_DATA/final_all_data/final_test.json",
        ]))

    # corpus.extend(
    #     load_cail2020_ydlj_corpus([
    #         "../cail_raw_data/2020/ydlj_small_data/train.json",
    #         "../cail_raw_data/2020/ydlj_big_data/train.json",
    #     ]))

    # corpus.extend(
    #     load_cail2021_aqbq_corpus([
    #         "../cail_raw_data/2021/案情标签_第一阶段/aqbq/train.json",
    #     ]))

    corpus.extend(
        load_cail2021_aljs_candidate_corpus(
            "../cail_raw_data/2021/类案检索_第一阶段/small/candidates/"
        ))

    corpus.extend(
        load_cail2021_ydlj_corpus([
            "../cail_raw_data/2021/阅读理解_第一阶段/ydlj_cjrc3.0_small_train.json"
        ]))

    corpus.extend(
        load_cail2021_xxcq_corpus([
            "../cail_raw_data/2021/信息抽取_第二阶段/xxcq_mid.json",
        ]))
    
    # 保留句子长度超过`min_length`的
    corpus = list(filter(lambda x: len(x) > args.min_length and len(x) < args.max_length, corpus))

    # 统计
    lengths = list(map(len, corpus))
    length_counter = Counter(lengths)
    num_corpus = len(corpus)
    print(f"{sys._getframe().f_code.co_name} #{len(corpus)}")
    # corpus = sorted(corpus, key=lambda x: -len(x))      # for debug

    # 保存
    random.shuffle(corpus)
    corpus = list(map(lambda x: x + "\n", corpus))
    with open(os.path.join(args.output_dir, "corpus.txt"), "w", encoding="utf-8") as f:
        f.writelines(corpus)
    
    # with open(os.path.join(args.output_dir, "corpus.txt"), "r", encoding="utf-8") as f:
    #     corpus = f.readlines()
    # corpus_train_tiny = corpus[:1000]
    # corpus_valid_tiny = corpus[1000:1200]
    # with open(os.path.join(args.output_dir, "corpus.train.tiny.txt"), "w", encoding="utf-8") as f:
    #     f.writelines(corpus_train_tiny)
    # with open(os.path.join(args.output_dir, "corpus.valid.tiny.txt"), "w", encoding="utf-8") as f:
    #     f.writelines(corpus_valid_tiny)

    if args.train_ratio is not None:
        num_corpus_train = int(num_corpus * args.train_ratio)
        corpus_train = corpus[: num_corpus_train]
        corpus_valid = corpus[num_corpus_train: ]
        with open(os.path.join(args.output_dir, "corpus.train.txt"), "w", encoding="utf-8") as f:
            f.writelines(corpus_train)
        with open(os.path.join(args.output_dir, "corpus.valid.txt"), "w", encoding="utf-8") as f:
            f.writelines(corpus_valid)

if __name__ == '__main__':
    parser = ArgumentParser()
    # parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--output_dir", type=str, default="../cail_processed_data/")
    parser.add_argument("--min_length", type=int, default=20)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--train_ratio", type=float, default=None)
    parser.add_argument("--seed", default=42, type=int, help="Seed.")
    args = parser.parse_args()

    main(args)
