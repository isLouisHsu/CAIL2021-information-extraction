import os
import json
import torch
import random
import numpy as np

def seed_everything(seed=None, reproducibility=True):
    '''
    init random seed for random functions in numpy, torch, cuda and cudnn
    Args:
        seed (int): random seed
        reproducibility (bool): Whether to require reproducibility
    '''
    if seed is None:
        seed = int(_select_seed_randomly())
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

LABEL_MEANING_MAP = {
    "NHCS": "犯罪嫌疑人",
    "NHVI": "受害人",
    "NCSM": "被盗货币",
    "NCGV": "物品价值",
    "NCSP": "盗窃获利",
    "NASI": "被盗物品",
    "NATS": "作案工具",
    "NT": "时间",
    "NS": "地点",
    "NO": "组织机构",
}

def load_raw(filepath):
    raw = []
    with open(os.path.join(filepath), "r") as f:
        for line in f.readlines():
            r = dict()
            line = json.loads(line)
            context = line["context"]
            r["id"] = line["id"]
            r["text"] = context
            r["sent_start"] = 0
            r["sent_end"] = len(context)
            r["entities"] = []
            if "entities" in line:
                for entity in line["entities"]:
                    for span in entity["span"]:
                        start, end = span.split(";")
                        start, end = int(start), int(end)
                        r["entities"].append((
                            entity["label"], 
                            start, end - 1,
                            context[start: end]
                        ))
            raw.append(r)
    return raw

def save_samples(filename, samples):
    with open(filename, "w") as f:
        for sample in samples:
            sample = json.dumps(sample, ensure_ascii=False) + "\n"
            f.write(sample)

if __name__ == "__main__":
    load_raw("./data/信息抽取_第一阶段/xxcq_small.json")