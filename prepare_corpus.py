import os
from argparse import ArgumentParser

def get_xxcq_corpus():
    """ 信息抽取 """
    ...

def get_sfzy_corpus():
    """ 司法摘要 """
    ...

def get_sfks_corpus():
    """ 司法考试 """
    ...

def get_aqbq_corpus():
    """ 案情标签 """
    ...

def get_aljs_corpus():
    """ 案类检索 """
    ...

def get_bllj_corpus():
    """ 辩论理解 """
    ...

def get_ydlj_corpus():
    """ 阅读理解 """
    ...

def main(args):
    args.output_dir = os.path.join(args.output_dir, f"mlm-seed{args.seed}")
    
    ...

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--output_dir", type=str, default="data/")
    parser.add_argument("--seed", default=42, type=int, help="Seed.")
    args = parser.parse_args()

    main(args)
