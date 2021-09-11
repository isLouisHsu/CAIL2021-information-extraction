import os
import json
import torch
import random
import numpy as np
from collections import Counter

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
MEANING_LABEL_MAP = {v: k for k, v in LABEL_MEANING_MAP.items()}

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

def count_entity_labels(samples):
    labels = []
    for sample in samples:
        for label, *other in sample["entities"]:
            labels.append(LABEL_MEANING_MAP[label])
    counter = Counter(labels)
    return counter

def save_samples(filename, samples):
    with open(filename, "w") as f:
        for sample in samples:
            sample = json.dumps(sample, ensure_ascii=False) + "\n"
            f.write(sample)

def add_context(ordered_samples, context_window):
    if context_window <= 0:
        return ordered_samples

    samples = copy.deepcopy(ordered_samples)
    for i in range(len(samples)):
        if i == 0: continue
        text = samples[i]["text"]
        add_left = (context_window-len(text)) // 2
        add_right = (context_window-len(text)) - add_left
        sent_start, sent_end = samples[i]["sent_start"], samples[i]["sent_end"]

        # add left context
        j = i - 1
        while j >= 0 and add_left > 0:
            context_to_add = samples[j]["text"][-add_left:]
            text = context_to_add + text
            add_left -= len(context_to_add)
            sent_start += len(context_to_add)
            sent_end += len(context_to_add)
            j -= 1

        # add right context
        j = i + 1
        while j < len(samples) and add_right > 0:
            context_to_add = samples[j]["text"][:add_right]
            text = text + context_to_add
            add_right -= len(context_to_add)
            j += 1
        
        # adjust entities
        entities = []
        for label, start, end, span_text in samples[i]["entities"]:
            start += sent_start; end += sent_start
            span_text_new = text[start: end + 1]
            assert span_text_new == span_text, "Error"
            entities.append((label, start, end, span_text))
        
        samples[i]["text"] = text
        samples[i]["sent_start"] = sent_start
        samples[i]["sent_end"] = sent_end
        samples[i]["entities"] = entities
    return samples

def get_ner_tags(entities, seq_len):
    ner_tags = ["O"] * seq_len
    for entity in entities:
        t, s, e = entity[:3]
        if s < 0 or s >= seq_len or e < 0 or e >= seq_len \
            or s > e or ner_tags[s] != "O" or ner_tags[e] != "O":
            continue
        ner_tags[s] = f"B-{t}"
        for i in range(s + 1, e + 1):
            ner_tags[i] = f"I-{t}"
    return ner_tags

if __name__ == "__main__":
    load_raw("./data/信息抽取_第一阶段/xxcq_small.json")