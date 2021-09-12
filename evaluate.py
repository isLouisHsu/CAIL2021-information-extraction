import os
import sys
import json
from collections import Counter
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from utils import LABEL_MEANING_MAP

plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False

def score(ground_truth, prediction, labels=None):
    ground_truth_num = 0
    prediction_num = 0
    tp = 0

    for id, ground_truth_data in ground_truth.items():
        try:
            pred_data = prediction[id]
        except KeyError:
            continue
        ground_truth_entities_dict = {}
        for entitie in ground_truth_data['entities']:
            if labels is not None and entitie["label"] not in labels:
                continue
            ground_truth_num += len(entitie['span'])
            ground_truth_entities_dict[entitie['label']] = entitie['span']
        pred_entities_dict = {}
        for entitie in pred_data['entities']:
            if labels is not None and entitie["label"] not in labels:
                continue
            prediction_num += len(entitie['span'])
            pred_entities_dict[entitie['label']] = entitie['span']
        for label in ground_truth_entities_dict.keys():
            tp += len(set(ground_truth_entities_dict[label]).intersection(set(pred_entities_dict[label])))

    try:
        p = tp / prediction_num
        r = tp / ground_truth_num
        f = 2 * p * r / ( p + r )
        score = {"p": p, "r": r, "f": f}
    except ZeroDivisionError as e:
        score = {"p": -1, "r": -1, "f": -1}
    return score


def get_scores(ground_truth_path, output_path): 
    ground_truth = {}
    prediction = {}
    with open(ground_truth_path, 'r', encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            id = data['id']
            data.pop('id')        
            ground_truth[id] = data
    with open(output_path, 'r', encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            id = data['id']
            data.pop('id')
            prediction[id] = data
    scores = dict()
    scores["avg"] = score(ground_truth, prediction)
    for label in LABEL_MEANING_MAP.keys():
        scores[label] = score(ground_truth, prediction, [label])
    return scores

def analyze_error(ground_truth_path, output_path):
    get_position = lambda x: "-".join(x.split("-")[:2])
    get_content = lambda x: "-".join(x.split("-")[2:])
    get_label = lambda x: x.split("-")[-1]
    ground_truth = {}
    prediction = {}
    with open(ground_truth_path, 'r', encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            id = data['id']
            data.pop('id')        
            ground_truth[id] = data
    with open(output_path, 'r', encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            id = data['id']
            data.pop('id')
            prediction[id] = data
    ground_truth_id_text_map = dict()
    ground_truth_entities = []
    for id, ground_truth_data in ground_truth.items():
        text = ground_truth_data["text"]
        ground_truth_id_text_map[id] = text
        for entity in ground_truth_data["entities"]:
            label = LABEL_MEANING_MAP[entity["label"]]
            for span in entity["span"]:
                b, e = [int(i) for i in span.split(";")]
                ground_truth_entities.append(f"{id}-{span}-{text[b: e]}-{label}")
    ground_truth_entities = sorted(ground_truth_entities, key=get_position)
    prediction_entities = []
    for id, prediction_data in prediction.items():
        text = ground_truth_id_text_map[id]
        for entity in prediction_data["entities"]:
            label = LABEL_MEANING_MAP[entity["label"]]
            for span in entity["span"]:
                b, e = [int(i) for i in span.split(";")]
                prediction_entities.append(f"{id}-{span}-{text[b: e]}-{label}")
    prediction_entities = sorted(prediction_entities, key=get_position)
    
    # in_gt_not_in_pred = sorted(set(ground_truth_entities) - set(prediction_entities))
    # in_pred_not_in_gt = sorted(set(prediction_entities) - set(ground_truth_entities))
    ground_truth_positions_content_map = {get_position(entity): get_content(entity) 
        for entity in ground_truth_entities}
    prediction_positions_content_map = {get_position(entity): get_content(entity) 
        for entity in prediction_entities}
    ground_truth_positions_label_map = {get_position(entity): get_label(entity) 
        for entity in ground_truth_entities}
    prediction_positions_label_map = {get_position(entity): get_label(entity) 
        for entity in prediction_entities}
    ## 第一类错误：定位错误，未识别到标注实体
    location_missed = set(ground_truth_positions_content_map.keys()) - set(prediction_positions_content_map.keys())
    location_missed = sorted(location_missed)
    location_missed = [k + "-" + ground_truth_positions_content_map[k] for k in location_missed]
    location_missed = sorted(location_missed)
    ## 第二类错误：定位错误，识别到未标注实体
    location_found = set(prediction_positions_content_map.keys()) - set(ground_truth_positions_content_map.keys())
    location_found = sorted(location_found)
    location_found = [k + "-" + prediction_positions_content_map[k] for k in location_found]
    location_found = sorted(location_found)
    ## 第三类错误：定位准确但分类错误
    itersection_positions = set(ground_truth_positions_content_map.keys()) \
        .intersection(prediction_positions_content_map.keys())
    label_error = [k + "-" + ground_truth_positions_content_map[k].split("-")[0] + "-" + \
            ground_truth_positions_label_map[k] + "-" + prediction_positions_label_map[k] 
        for k in itersection_positions if ground_truth_positions_label_map[k] != prediction_positions_label_map[k]]
    # 混淆矩阵
    y_true = [ground_truth_positions_label_map[k] for k in itersection_positions]
    y_pred = [prediction_positions_label_map[k] for k in itersection_positions]
    labels = list(LABEL_MEANING_MAP.values())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels).plot()
    label_error_type_count_map = Counter(["-".join(e.split("-")[-2:]) for e in label_error])
    # 保存
    for list_name in ["ground_truth_entities", "prediction_entities", 
            "location_missed", "location_found", "label_error"]:
        with open(os.path.join("tmp", list_name + ".txt"), "w") as f:
            for line in locals()[list_name]:
                f.write(line + "\n")
    print(label_error_type_count_map)
    print(labels)
    plt.savefig(os.path.join("tmp", "cm.jpg"))

if __name__ == '__main__':
    # ground_truth_path, output_path = sys.argv[1], sys.argv[2]
    ground_truth_path, output_path = "data/ner-ctx0-5fold-seed42/dev.gt.all.json", "output.json"
    for label, score in get_scores(ground_truth_path, output_path).items():
        print(LABEL_MEANING_MAP.get(label, label))
        print(score)
    analyze_error(ground_truth_path, output_path)

    # for label, score in get_scores(
    #     "./data/ner-ctx0-5fold-seed42/dev.gt.0.json", 
    #     "output/ner-cail_ner-bert_span-baseline-fold0-42/test_prediction.json"
    # ).items():
    #     print(LABEL_MEANING_MAP.get(label, label))
    #     print(score)
    # analyze_error(
    #     "./data/ner-ctx0-5fold-seed42/dev.gt.0.json", 
    #     "output/ner-cail_ner-bert_span-baseline-fold0-42/test_prediction.json"
    # )
