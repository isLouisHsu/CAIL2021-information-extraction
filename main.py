#coding: utf-8
import os
import json
import utils
from run_span import NerArgumentParser

infile = "/input/input.json"
outfile = "/output/output.json"

def main():

    json_file = "./args/bert_span-baseline-pred.json"
    parser = NerArgumentParser()
    args = parser.parse_args_from_json(json_file=json_file)
    
    version = args.version
    data_dir = args.data_dir
    model_type = args.model_type
    dataset_name = args.dataset_name
    seed = args.seed

    raw_samples = utils.load_raw(infile)
    utils.save(os.path.join(data_dir, "test.json"), raw_samples)
    os.system(f"sudo /home/user/miniconda/bin/python3 run_span.py {json_file}")
    os.system(f"sudo mv ./output/ner-{dataset_name}-{model_type}-{version}-{seed}/test_prediction.json {outfile}")
    
    # local test
    # os.system(f"python run_span.py {json_file}")
    # os.system(f"sudo mv ./output/ner-{dataset_name}-{model_type}-{version}-{seed}/test_prediction.json ./output.json")

if __name__ == '__main__':
    main()
