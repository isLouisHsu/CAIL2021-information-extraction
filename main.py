#coding: utf-8
import os
import json
import utils
from run_span import NerArgumentParser

infile = "/input/input.json"
outfile = "/output/output.json"

def main():

    context_window = 510
    json_file = "output/ner-cail_ner-bert_span-baseline-42/training_args.json"
    
    parser = NerArgumentParser()
    args = parser.parse_args_from_json(json_file=json_file)
    args.test_file = "test.json"
    args.do_train = False
    args.do_eval = False
    args.do_test = True
    args.per_gpu_eval_batch_size = 1
    args.fp16 = False
    parser.save_args_to_json("./args/pred.json", args)
    
    version = args.version
    data_dir = args.data_dir
    model_type = args.model_type
    dataset_name = args.dataset_name
    seed = args.seed

    # upload
    raw_samples = utils.load_raw(infile)
    raw_samples = utils.add_context(raw_samples, context_window)
    utils.save_samples(os.path.join(data_dir, "test.json"), raw_samples)
    
    os.system(f"sudo /home/user/miniconda/bin/python3 run_span.py ./args/pred.json")
    os.system(f"sudo cp ./output/ner-{dataset_name}-{model_type}-{version}-{seed}/test_prediction.json {outfile}")

    # # local test
    # os.system(f"python run_span.py {json_file}")
    # os.system(f"mv ./output/ner-{dataset_name}-{model_type}-{version}-{seed}/test_prediction.json ./output.json")

    # with open(f"./output/ner-{dataset_name}-{model_type}-{version}-{seed}/test_prediction.json", "r") as f:
    #     content = f.read()
    # with open(outfile, "w") as f:
    #     f.write(content)

if __name__ == '__main__':
    main()
