# CAIL2021——司法文本信息抽取

[报名链接](http://cail.cipsc.org.cn/)，方案详细介绍见[中国法律智能技术评测(CAIL2021)：信息抽取(Rank2) - LOUIS' BLOG](https://louishsu.xyz/2021/10/22/%E4%B8%AD%E5%9B%BD%E6%B3%95%E5%BE%8B%E6%99%BA%E8%83%BD%E6%8A%80%E6%9C%AF%E8%AF%84%E6%B5%8B(CAIL2021)%EF%BC%9A%E4%BF%A1%E6%81%AF%E6%8A%BD%E5%8F%96(Rank2).html)

## 数据说明

本次任务所使用的数据集主要来自于网络公开的若干罪名起诉意见书，总计7500余条样本，10类相关业务相关实体，分别为犯罪嫌疑人、受害人、作案工具、被盗物品、被盗货币、物品价值、盗窃获利、时间、地点、组织机构。其中第一阶段数据集包含约1700条样本。每条样本中均包含任意数目的实体。考虑到多类罪名案件交叉的复杂性，本次任务仅涉及盗窃罪名的相关信息抽取。

针对本次任务，我们会提供包含案件情节描述的陈述文本，选手需要识别出文本中的关键信息实体，并按照规定格式返回结果。

发放的文件为``xxcq_small.json``，为字典列表，字典包含字段为：

- ``id``：案例中句子的唯一标识符。
- ``context``：句子内容，抽取自起诉意见书的事实描述部分。
- ``entities``：句子所包含的实体列表。
- ``label``：实体标签名称。
- ``span``：实体在``context``中的起止位置。

其中``label``的十种实体类型分别为：

|label|含义|
|---|---|
|NHCS|犯罪嫌疑人|
|NHVI|受害人|
|NCSM|被盗货币|
|NCGV|物品价值|
|NCSP|盗窃获利|
|NASI|被盗物品|
|NATS|作案工具|
|NT|时间|
|NS|地点|
|NO|组织机构|

## 程序运行

#### 准备数据
``` sh
$ tree ../cail_raw_data/ -d
../cail_raw_data/
├── 2018
│   └── CAIL2018_ALL_DATA
│       └── final_all_data
│           ├── exercise_contest
│           ├── first_stage
│           └── restData
├── 2020
│   ├── ydlj_big_data
│   └── ydlj_small_data
└── 2021
    ├── 信息抽取_第一阶段
    ├── 信息抽取_第二阶段
    ├── 案情标签_第一阶段
    │   ├── aqbq
    │   └── __MACOSX
    │       └── aqbq
    ├── 类案检索_第一阶段
    │   ├── __MACOSX
    │   │   └── small
    │   │       └── candidates
    │   └── small
    │       └── candidates
    │           ├── 1325
    │           ├── 1355
    │           ├── 1405
    │           ├── 1430
    │           ├── 1972
    │           ├── 1978
    │           ├── 2132
    │           ├── 2143
    │           ├── 221
    │           ├── 2331
    │           ├── 2361
    │           ├── 2373
    │           ├── 259
    │           ├── 3228
    │           ├── 3342
    │           ├── 3746
    │           ├── 3765
    │           ├── 4738
    │           ├── 4794
    │           └── 4829
    └── 阅读理解_第一阶段

43 directories
```

#### 领域预训练
``` sh
# 生成预训练语料
python prepare_corpus.py \
    --output_dir=../cail_processed_data/ \
    --min_length=30 \
    --max_length=256 \
    --seed=42

# 分词用于wwm
python run_chinese_ref.py \
    --file_name=../cail_processed_data/mlm-minlen30-maxlen256-seed42/corpus.txt \
    --ltp=/home/louishsu/NewDisk/Garage/weights/ltp/base1.tgz \
    --bert=/home/louishsu/NewDisk/Garage/weights/transformers/nezha-cn-base/ \
    --save_path=../cail_processed_data/mlm-minlen30-maxlen256-seed42/ref.txt

# 预训练
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
```

#### 信息抽取微调

``` sh
# 生成数据
python prepare_data.py \
    --data_files ./data/信息抽取_第二阶段/xxcq_mid.json \
    --context_window 0 \
    --n_splits 5 \
    --output_dir data/ \
    --seed 42

# 多折模型微调
for k in 0 1 2 3 4
do
python run_span.py \
    --version=nezha-legal-fgm1.0-lsr0.1-fold${k} \
    --data_dir=./data/ner-ctx0-5fold-seed42/ \
    --train_file=train.${k}.json \
    --dev_file=dev.${k}.json \
    --test_file=dev.${k}.json \
    --model_type=nezha_span \
    --model_name_or_path=/home/louishsu/NewDisk/Code/CAIL2021/nezha-legal-cn-base-wwm/ \
    --do_train \
    --overwrite_output_dir \
    --evaluate_during_training \
    --evaluate_each_epoch \
    --save_best_checkpoints \
    --max_span_length=40 \
    --width_embedding_dim=128 \
    --train_max_seq_length=512 \
    --eval_max_seq_length=512 \
    --do_lower_case \
    --per_gpu_train_batch_size=8 \
    --per_gpu_eval_batch_size=16 \
    --gradient_accumulation_steps=2 \
    --learning_rate=5e-5 \
    --other_learning_rate=1e-3 \
    --num_train_epochs=8.0 \
    --warmup_proportion=0.1 \
    --do_fgm --fgm_epsilon=1.0 \
    --loss_type=lsr --label_smooth_eps=0.1 \
    --seed=42
done

# 本地线下预测得到OOF，并计算得分
python main_local.py
```

注：可通过以下命令校验划分得到的数据集是否一致
``` sh
$ cd data/md5sum -c checksum
$ md5sum -c checksum
```

<!-- 
``` sh
$ for f in `ls`
> do
> md5sum $f >> checksum
> done
```
-->
