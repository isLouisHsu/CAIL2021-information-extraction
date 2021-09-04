#coding: utf-8
import os
import json
import utils
from run_span import NerArgumentParser

infile = "/input/input.json"
outfile = "/output/output.json"

# def main():

#     model_path = "./output/ner-cail_ner-bert_span-baseline-42/"
#     # json_file = "./args/bert_span-baseline.json"
#     json_file = os.path.join(model_path, "training_args.json")
#     parser = NerArgumentParser()
#     args = parser.parse_args_from_json(json_file=json_file)
#     args.test_file = "test.json"
#     args.do_train = False
#     args.do_eval = False
#     args.do_test = True
#     args.per_gpu_eval_batch_size = 1
#     args.fp16 = False
#     parser.save_args_to_json("./args/pred.json", args)
#     os.system(f"rm -rf {os.path.join(model_path, 'test_prediction.json')}")
    
#     version = args.version
#     data_dir = args.data_dir
#     model_type = args.model_type
#     dataset_name = args.dataset_name
#     seed = args.seed

#     # upload
#     raw_samples = utils.load_raw(infile)
#     utils.save_samples(os.path.join(data_dir, "test.json"), raw_samples)
#     os.system(f"sudo /home/user/miniconda/bin/python3 run_span.py ./args/pred.json")
#     os.system(f"sudo cp ./output/ner-{dataset_name}-{model_type}-{version}-{seed}/test_prediction.json {outfile}")
#     exit(0)

#     # local test
#     os.system(f"python run_span.py ./args/pred.json")
#     os.system(f"mv ./output/ner-{dataset_name}-{model_type}-{version}-{seed}/test_prediction.json ./output.json")

# TODO: kfold

if __name__ == '__main__':
    main()
