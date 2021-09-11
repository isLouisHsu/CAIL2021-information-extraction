#!/bin/bash

# # >>> 第一阶段 >>>
# python prepare_data.py \
#     --data_files ./data/信息抽取_第一阶段/xxcq_small.json \
#     --context_window 0 \
#     --n_splits 5 \
#     --output_dir data/ \
#     --seed 42
# # INFO:root:Saving to data/ner-ctx0-5fold-seed42
# # INFO:root:Number of raw samples: 2277
# # INFO:root:Fold[0/5] Number of training data: 1821, number of dev data: 456
# # INFO:root:Counter({'犯罪嫌疑人': 2247, '被盗物品': 2028, '地点': 1248, '受害人': 1045, '时间': 975, '物品价值': 748, '被盗货币': 332, '组织机构': 263, '作案工具': 242, '盗窃获利': 146})
# # INFO:root:Counter({'犯罪嫌疑人': 688, '被盗物品': 527, '地点': 332, '受害人': 254, '时间': 251, '物品价值': 199, '被盗货币': 86, '组织机构': 83, '作案工具': 52, '盗窃获利': 40})
# # INFO:root:Fold[1/5] Number of training data: 1821, number of dev data: 456
# # INFO:root:Counter({'犯罪嫌疑人': 2362, '被盗物品': 2048, '地点': 1293, '受害人': 1022, '时间': 985, '物品价值': 771, '被盗货币': 330, '组织机构': 286, '作案工具': 252, '盗窃获利': 139})
# # INFO:root:Counter({'犯罪嫌疑人': 573, '被盗物品': 507, '地点': 287, '受害人': 277, '时间': 241, '物品价值': 176, '被盗货币': 88, '组织机构': 60, '盗窃获利': 47, '作案工具': 42})
# # INFO:root:Fold[2/5] Number of training data: 1822, number of dev data: 455
# # INFO:root:Counter({'犯罪嫌疑人': 2361, '被盗物品': 2075, '地点': 1266, '受害人': 1044, '时间': 997, '物品价值': 771, '被盗货币': 326, '组织机构': 281, '作案工具': 254, '盗窃获利': 152})
# # INFO:root:Counter({'犯罪嫌疑人': 574, '被盗物品': 480, '地点': 314, '受害人': 255, '时间': 229, '物品价值': 176, '被盗货币': 92, '组织机构': 65, '作案工具': 40, '盗窃获利': 34})
# # INFO:root:Fold[3/5] Number of training data: 1822, number of dev data: 455
# # INFO:root:Counter({'犯罪嫌疑人': 2388, '被盗物品': 2041, '地点': 1255, '受害人': 1064, '时间': 976, '物品价值': 757, '被盗货币': 344, '组织机构': 278, '作案工具': 217, '盗窃获利': 157})
# # INFO:root:Counter({'犯罪嫌疑人': 547, '被盗物品': 514, '地点': 325, '时间': 250, '受害人': 235, '物品价值': 190, '作案工具': 77, '被盗货币': 74, '组织机构': 68, '盗窃获利': 29})
# # INFO:root:Fold[4/5] Number of training data: 1822, number of dev data: 455
# # INFO:root:Counter({'犯罪嫌疑人': 2382, '被盗物品': 2028, '地点': 1258, '受害人': 1021, '时间': 971, '物品价值': 741, '被盗货币': 340, '组织机构': 276, '作案工具': 211, '盗窃获利': 150})
# # INFO:root:Counter({'犯罪嫌疑人': 553, '被盗物品': 527, '地点': 322, '受害人': 278, '时间': 255, '物品价值': 206, '作案工具': 83, '被盗货币': 78, '组织机构': 70, '盗窃获利': 36})
# for k in 0 1 2 3 4
# do
# python run_span.py \
#     --version=baseline-fold${k} \
#     --data_dir=./data/ner-ctx0-5fold-seed42/ \
#     --train_file=train.${k}.json \
#     --dev_file=dev.${k}.json \
#     --test_file=dev.${k}.json \
#     --model_type=bert_span \
#     --model_name_or_path=/home/louishsu/NewDisk/Garage/weights/transformers/chinese-roberta-wwm/ \
#     --do_train \
#     --do_eval \
#     --do_predict \
#     --overwrite_output_dir \
#     --evaluate_during_training \
#     --evaluate_each_epoch \
#     --save_best_checkpoints \
#     --max_span_length=35 \
#     --width_embedding_dim=128 \
#     --train_max_seq_length=512 \
#     --eval_max_seq_length=512 \
#     --do_lower_case \
#     --per_gpu_train_batch_size=12 \
#     --per_gpu_eval_batch_size=24 \
#     --gradient_accumulation_steps=2 \
#     --learning_rate=2e-5 \
#     --other_learning_rate=1e-4 \
#     --num_train_epochs=8.0 \
#     --warmup_proportion=0.1 \
#     --seed=42
# python evaluate.py \
#     ./data/ner-ctx0-5fold-seed42/dev.gt.${k}.json \
#     output/ner-cail_ner-bert_span-baseline-fold${k}-42/test_prediction.json
# done
# # <<< 第一阶段 <<<

# >>> 第二阶段 >>>
# python prepare_data.py \
#     --data_files ./data/信息抽取_第二阶段/xxcq_mid.json \
#     --context_window 0 \
#     --n_splits 5 \
#     --output_dir data/ \
#     --seed 42
# INFO:root:Saving to data/ner-ctx0-5fold-seed42
# INFO:root:Number of raw samples: 5247
# INFO:root:Fold[0/5] Number of training data: 4197, number of dev data: 1050
# INFO:root:Counter({'犯罪嫌疑人': 5229, '被盗物品': 4636, '地点': 2798, '受害人': 2514, '时间': 2240, '物品价值': 1682, '被盗货币': 742, '组织机构': 664, '作案工具': 608, '盗窃获利': 363})
# INFO:root:Counter({'犯罪嫌疑人': 1234, '被盗物品': 1145, '地点': 719, '受害人': 594, '时间': 525, '物品价值': 408, '被盗货币': 173, '组织机构': 142, '作案工具': 127, '盗窃获利': 118})
# INFO:root:Fold[1/5] Number of training data: 4197, number of dev data: 1050
# INFO:root:Counter({'犯罪嫌疑人': 5219, '被盗物品': 4641, '地点': 2870, '受害人': 2514, '时间': 2224, '物品价值': 1692, '被盗货币': 749, '组织机构': 631, '作案工具': 554, '盗窃获利': 392})
# INFO:root:Counter({'犯罪嫌疑人': 1244, '被盗物品': 1140, '地点': 647, '受害人': 594, '时间': 541, '物品价值': 398, '作案工具': 181, '组织机构': 175, '被盗货币': 166, '盗窃获利': 89})
# INFO:root:Fold[2/5] Number of training data: 4198, number of dev data: 1049
# INFO:root:Counter({'犯罪嫌疑人': 5167, '被盗物品': 4654, '地点': 2811, '受害人': 2404, '时间': 2195, '物品价值': 1668, '被盗货币': 726, '组织机构': 630, '作案工具': 585, '盗窃获利': 385})
# INFO:root:Counter({'犯罪嫌疑人': 1296, '被盗物品': 1127, '地点': 706, '受害人': 704, '时间': 570, '物品价值': 422, '被盗货币': 189, '组织机构': 176, '作案工具': 150, '盗窃获利': 96})
# INFO:root:Fold[3/5] Number of training data: 4198, number of dev data: 1049
# INFO:root:Counter({'犯罪嫌疑人': 5086, '被盗物品': 4614, '地点': 2812, '受害人': 2462, '时间': 2209, '物品价值': 1682, '被盗货币': 698, '组织机构': 651, '作案工具': 629, '盗窃获利': 382})
# INFO:root:Counter({'犯罪嫌疑人': 1377, '被盗物品': 1167, '地点': 705, '受害人': 646, '时间': 556, '物品价值': 408, '被盗货币': 217, '组织机构': 155, '作案工具': 106, '盗窃获利': 99})
# INFO:root:Fold[4/5] Number of training data: 4198, number of dev data: 1049
# INFO:root:Counter({'犯罪嫌疑人': 5151, '被盗物品': 4579, '地点': 2777, '受害人': 2538, '时间': 2192, '物品价值': 1636, '被盗货币': 745, '组织机构': 648, '作案工具': 564, '盗窃获利': 402})
# INFO:root:Counter({'犯罪嫌疑人': 1312, '被盗物品': 1202, '地点': 740, '时间': 573, '受害人': 570, '物品价值': 454, '作案工具': 171, '被盗货币': 170, '组织机构': 158, '盗窃获利': 79})
for k in 0 1 2 3 4
do
python run_span.py \
    --version=baseline-fold${k} \
    --data_dir=./data/ner-ctx0-5fold-seed42/ \
    --train_file=train.${k}.json \
    --dev_file=dev.${k}.json \
    --test_file=dev.${k}.json \
    --model_type=bert_span \
    --model_name_or_path=/home/louishsu/NewDisk/Garage/weights/transformers/chinese-roberta-wwm/ \
    --do_train \
    --do_eval \
    --do_predict \
    --overwrite_output_dir \
    --evaluate_during_training \
    --evaluate_each_epoch \
    --save_best_checkpoints \
    --max_span_length=50 \
    --width_embedding_dim=128 \
    --train_max_seq_length=512 \
    --eval_max_seq_length=512 \
    --do_lower_case \
    --per_gpu_train_batch_size=12 \
    --per_gpu_eval_batch_size=24 \
    --gradient_accumulation_steps=2 \
    --learning_rate=2e-5 \
    --other_learning_rate=1e-4 \
    --num_train_epochs=8.0 \
    --warmup_proportion=0.1 \
    --seed=42
python evaluate.py \
    ./data/ner-ctx0-5fold-seed42/dev.gt.${k}.json \
    output/ner-cail_ner-bert_span-baseline-fold${k}-42/test_prediction.json
done

# INFO:root:Counter({'犯罪嫌疑人': 1312, '被盗物品': 1202, '地点': 740, '时间': 573, '受害人': 570, '物品价值': 454, '作案工具': 171, '被盗货币': 170, '组织机构': 158, '盗窃获利': 79})
for k in 0 1 2 3 4
do
python run_span.py \
    --version=legal_electra_base-fold${k} \
    --data_dir=./data/ner-ctx0-5fold-seed42/ \
    --train_file=train.${k}.json \
    --dev_file=dev.${k}.json \
    --test_file=dev.${k}.json \
    --model_type=bert_span \
    --model_name_or_path=/home/louishsu/NewDisk/Garage/weights/transformers/hfl_chinese-legal-electra-base-discriminator/ \
    --do_train \
    --do_eval \
    --do_predict \
    --overwrite_output_dir \
    --evaluate_during_training \
    --evaluate_each_epoch \
    --save_best_checkpoints \
    --max_span_length=50 \
    --width_embedding_dim=128 \
    --train_max_seq_length=512 \
    --eval_max_seq_length=512 \
    --do_lower_case \
    --per_gpu_train_batch_size=12 \
    --per_gpu_eval_batch_size=24 \
    --gradient_accumulation_steps=2 \
    --learning_rate=2e-5 \
    --other_learning_rate=1e-4 \
    --num_train_epochs=8.0 \
    --warmup_proportion=0.1 \
    --seed=42
python evaluate.py \
    ./data/ner-ctx0-5fold-seed42/dev.gt.${k}.json \
    output/ner-cail_ner-bert_span-baseline-fold${k}-42/test_prediction.json
done
# <<< 第二阶段 <<<