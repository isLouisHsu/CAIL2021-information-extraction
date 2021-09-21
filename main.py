#coding: utf-8
import os
import json
import utils
import torch
import numpy as np
from argparse import ArgumentParser
from run_span import NerArgumentParser, CailNerProcessor, predict_decode_batch

infile = "/input/input.json"
outfile = "/output/output.json"

def main():
    parser = ArgumentParser()
    parser.add_argument("--local_debug", action="store_true", default=False)
    parser.add_argument("--test_file", type=str, default="dev.all.json")
    run_args = parser.parse_args()
    local_debug = run_args.local_debug

    # WARNING：以下配置需要在提交前指定！！！
    # version = "baseline"
    # model_type = "bert_span"
    # dataset_name = "cail_ner"
    # n_splits = 5
    # seed=42
    # --------------------------
    # version = "rdrop0.1-fgm1.0"
    # model_type = "bert_span"
    # dataset_name = "cail_ner"
    # n_splits = 5
    # seed=42
    # --------------------------
    # version = "nezha_rdrop0.1-fgm1.0"
    # model_type = "nezha_span"
    # dataset_name = "cail_ner"
    # n_splits = 5
    # seed=42
    # --------------------------
    # version = "nezha-fgm1.0"
    # model_type = "nezha_span"
    # dataset_name = "cail_ner"
    # n_splits = 5
    # seed=42
    # --------------------------
    version = "nezha-rdrop0.1-fgm1.0-aug_ctx0.15"
    model_type = "nezha_span"
    dataset_name = "cail_ner"
    n_splits = 5
    seed=42

    test_examples = None
    test_batches = None
    for k in range(n_splits):
        model_path = f"./output/ner-{dataset_name}-{model_type}-{version}-fold{k}-{seed}/"
        # 生成测试运行参数
        json_file = os.path.join(model_path, "training_args.json")
        parser = NerArgumentParser()
        args = parser.parse_args_from_json(json_file)
        args.do_train, args.do_eval, args.do_predict = False, False, True
        args.per_gpu_eval_batch_size = 1
        args.fp16 = False
        # 生成测试数据集
        if not local_debug:
            raw_samples = utils.load_raw(infile)
            utils.save_samples(os.path.join(args.data_dir, "test.json"), raw_samples)
            args.test_file = "test.json"
        else:
            args.test_file = run_args.test_file
        parser.save_args_to_json(f"./args/pred.{k}.json", args)
        # 确保目录下预测输出文件被清除
        os.system(f"rm -rf {os.path.join(model_path, 'test_*')}")

        if local_debug:
            # 线下预测测试
            os.system(f"python run_span.py ./args/pred.{k}.json")
        else:
            # 线上预测阶段
            os.system(f"sudo /home/user/miniconda/bin/python3 run_span.py ./args/pred.{k}.json")
        
        # 读取预测输出，并集成
        test_examples_ = torch.load(os.path.join(model_path, 'test_examples.pkl'))
        test_batches_ = torch.load(os.path.join(model_path, 'test_batches.pkl'))
        if test_examples is None:
            test_examples, test_batches = test_examples_, test_batches_
        else:
            for i, (batch, batch_) in enumerate(zip(test_batches, test_batches_)):
                test_batches[i]["logits"] = batch["logits"] + batch_["logits"]

    # 集成结果预测
    results = []
    for i, (example, batch) in enumerate(zip(test_examples, test_batches)):
        results.append(predict_decode_batch(example[1], batch, CailNerProcessor().id2label))
    # 保存结果
    output_predict_file = "output.json" if local_debug else outfile
    with open(output_predict_file, "w") as writer:
        for record in results:
            writer.write(json.dumps(record) + '\n')
    
    if local_debug:
        tmp = run_args.test_file.split(".")
        tmp.insert(1, "gt")
        gt_file = ".".join(tmp)
        os.system(f"python evaluate.py data/ner-ctx0-5fold-seed42/{gt_file} output.json")

if __name__ == '__main__':
    main()
