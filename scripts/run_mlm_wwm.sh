#!/bin/bash
###
 # @Description: 
 # @Version: 
 # @Author: louishsu
 # @Github: https://github.com/isLouisHsu
 # @E-mail: is.louishsu@foxmail.com
 # 
 # @Date: 2021-09-19 14:53:15
 # @LastEditTime: 2021-09-20 16:55:25
 # @LastEditors: louishsu
 # @FilePath: \CAIL2021-information-extraction\scripts\run_mlm_wwm.sh
### 

# python prepare_corpus.py \
#     --output_dir=../cail_processed_data/ \
#     --min_length=20 \
#     --max_length=256 \
#     --train_ratio=0.9 \
#     --seed=42

# for data_type in train valid
# do
# python run_chinese_ref.py \
#     --file_name=../cail_processed_data/mlm-minlen20-maxlen256-seed42/corpus.${data_type}.txt \
#     --ltp=/home/louishsu/NewDisk/Garage/weights/ltp/base1.tgz \
#     --bert=/home/louishsu/NewDisk/Garage/weights/transformers/nezha-cn-base/ \
#     --save_path=../cail_processed_data/mlm-minlen20-maxlen256-seed42/ref.${data_type}.txt
# done

# export WANDB_DISABLED=true
# nohup python run_mlm_wwm.py \
#     --model_name_or_path=/home/louishsu/NewDisk/Garage/weights/transformers/nezha-cn-base/ \
#     --model_type=nezha \
#     --train_file=../cail_processed_data/mlm-minlen20-maxlen256-seed42/corpus.train.txt \
#     --validation_file=../cail_processed_data/mlm-minlen20-maxlen256-seed42/corpus.valid.txt \
#     --train_ref_file=../cail_processed_data/mlm-minlen20-maxlen256-seed42/ref.train.txt \
#     --validation_ref_file=../cail_processed_data/mlm-minlen20-maxlen256-seed42/ref.valid.txt \
#     --cache_dir=cache/ \
#     --overwrite_cache \
#     --max_seq_length=256 \
#     --preprocessing_num_workers=8 \
#     --mlm_probability=0.15 \
#     --output_dir=output/nezha-legal-cn-base-wwm/ \
#     --overwrite_output_dir \
#     --do_train --do_eval \
#     --warmup_steps=1000 \
#     --max_steps=10000 \
#     --evaluation_strategy=steps \
#     --eval_steps=500 \
#     --per_device_train_batch_size=32 \
#     --per_device_eval_batch_size=32 \
#     --gradient_accumulation_steps=2 \
#     --label_smoothing_factor=0.0 \
#     --learning_rate=5e-5 \
#     --weight_decay=0.01 \
#     --logging_dir=output/nezha-legal-cn-base-wwm/log/ \
#     --logging_strategy=steps \
#     --logging_steps=500 \
#     --save_strategy=steps \
#     --save_steps=500 \
#     --save_total_limit=10 \
#     --dataloader_num_workers=4 \
#     --seed=42 \
#     --fp16 \
# >> output/nezha-legal-cn-base-wwm.out &

# ================================================================================================
python prepare_corpus.py \
    --output_dir=../cail_processed_data/ \
    --min_length=30 \
    --max_length=256 \
    --seed=42

python run_chinese_ref.py \
    --file_name=../cail_processed_data/mlm-minlen30-maxlen256-seed42/corpus.txt \
    --ltp=/home/louishsu/NewDisk/Garage/weights/ltp/base1.tgz \
    --bert=/home/louishsu/NewDisk/Garage/weights/transformers/nezha-cn-base/ \
    --save_path=../cail_processed_data/mlm-minlen30-maxlen256-seed42/ref.txt

export WANDB_DISABLED=true
nohup python run_mlm_wwm.py \
    --model_name_or_path=/home/louishsu/NewDisk/Garage/weights/transformers/nezha-cn-base/ \
    --model_type=nezha \
    --train_file=../cail_processed_data/mlm-minlen30-maxlen256-seed42/corpus.txt \
    --train_ref_file=../cail_processed_data/mlm-minlen30-maxlen256-seed42/ref.txt \
    --cache_dir=cache/ \
    --overwrite_cache \
    --max_seq_length=256 \
    --preprocessing_num_workers=8 \
    --mlm_probability=0.15 \
    --output_dir=output/nezha-legal-cn-base-wwm/ \
    --overwrite_output_dir \
    --do_train \
    --warmup_steps=1500 \
    --max_steps=30000 \
    --per_device_train_batch_size=48 \
    --gradient_accumulation_steps=4 \
    --label_smoothing_factor=0.0 \
    --learning_rate=5e-5 \
    --weight_decay=0.01 \
    --logging_dir=output/nezha-legal-cn-base-wwm/log/ \
    --logging_strategy=steps \
    --logging_steps=1500 \
    --save_strategy=steps \
    --save_steps=1500 \
    --save_total_limit=10 \
    --dataloader_num_workers=4 \
    --seed=42 \
    --fp16 \
>> output/nezha-legal-cn-base-wwm.out &
