import os
import sys
import json
from collections import Counter
from utils import LABEL_MEANING_MAP

def get_score(ground_truth_path, output_path): 
    try:
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
                ground_truth_num += len(entitie['span'])
                ground_truth_entities_dict[entitie['label']] = entitie['span']
            pred_entities_dict = {}
            for entitie in pred_data['entities']:
                prediction_num += len(entitie['span'])
                pred_entities_dict[entitie['label']] = entitie['span']
            for label in ground_truth_entities_dict.keys():
                tp += len(set(ground_truth_entities_dict[label]).intersection(set(pred_entities_dict[label])))

        p = tp / prediction_num
        r = tp / ground_truth_num
        f = 2 * p * r / ( p + r )
            
        s1 = round(p * 100, 2)
        s2 = round(r * 100, 2)
        s3 = round(f * 100, 2)
        return {"p": s1, "r": s2, "f": s3}
    except Exception as e:
        return {"p": -1, "r": -1, "f": -1}

def analyze_error(ground_truth_path, output_path):
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
    prediction_entities = []
    for id, prediction_data in prediction.items():
        text = ground_truth_id_text_map[id]
        for entity in prediction_data["entities"]:
            label = LABEL_MEANING_MAP[entity["label"]]
            for span in entity["span"]:
                b, e = [int(i) for i in span.split(";")]
                prediction_entities.append(f"{id}-{span}-{text[b: e]}-{label}")
    
    # in_gt_not_in_pred = sorted(set(ground_truth_entities) - set(prediction_entities))
    # in_pred_not_in_gt = sorted(set(prediction_entities) - set(ground_truth_entities))
    get_position = lambda x: "-".join(x.split("-")[:2])
    get_content = lambda x: "-".join(x.split("-")[2:])
    get_label = lambda x: x.split("-")[-1]
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
    location_missed = [k + "-" + ground_truth_positions_content_map[k] for k in location_missed]
    location_missed = sorted(location_missed)
    ## 第二类错误：定位错误，识别到未标注实体
    location_found = set(prediction_positions_content_map.keys()) - set(ground_truth_positions_content_map.keys())
    location_found = [k + "-" + prediction_positions_content_map[k] for k in location_found]
    location_found = sorted(location_found)
    ## 第三类错误：定位准确但分类错误
    itersection_positions = set(ground_truth_positions_content_map.keys()) \
        .intersection(prediction_positions_content_map.keys())
    label_error = [k + "-" + ground_truth_positions_content_map[k] + \
            ground_truth_positions_label_map[k] + "->" + prediction_positions_label_map[k] 
        for k in itersection_positions if ground_truth_positions_label_map[k] != prediction_positions_label_map[k]]
    label_error_type_count_map = Counter(["-".join(e.split("-")[-2:]) for e in label_error])
    # 保存
    for list_name in ["location_missed", "location_found", "label_error"]:
        with open(os.path.join("tmp", list_name + ".txt"), "w") as f:
            for line in locals()[list_name]:
                f.write(line + "\n")
    print(label_error_type_count_map)

if __name__ == '__main__':
    # print(get_score(sys.argv[1], sys.argv[2]))
    # print(get_score(
    #     "./data/ner-ctx0-5fold-seed42/dev.gt.0.json", 
    #     "output/ner-cail_ner-bert_span-baseline-fold0-42/test_prediction.json"
    # ))
    analyze_error(
        "./data/ner-ctx0-5fold-seed42/dev.gt.0.json", 
        "output/ner-cail_ner-bert_span-baseline-fold0-42/test_prediction.json"
    )
