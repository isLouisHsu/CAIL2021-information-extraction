#!/bin/bash
# python run_span.py \
#     --version=baseline \
#     --device=cuda:0 \
#     --n_gpu=1 \
#     --task_name=ner \
#     --dataset_name=cail_ner \
#     --data_dir=./data/ner-ctx0-train0.8-seed42/ \
#     --train_file=train.json \
#     --dev_file=dev.json \
#     --test_file=dev.json \
#     --model_type=bert_span \
#     --model_name_or_path=/home/louishsu/NewDisk/Garage/weights/transformers/chinese-roberta-wwm/ \
#     --max_span_length=30 \
#     --width_embedding_dim=128 \
#     --train_max_seq_length=512 \
#     --eval_max_seq_length=512 \
#     --per_gpu_train_batch_size=16 \
#     --per_gpu_eval_batch_size=24 \
#     --gradient_accumulation_steps=2 \
#     --learning_rate=2e-5 \
#     --other_learning_rate=1e-3 \
#     --num_train_epochs=10.0 \
#     --do_train \
#     --do_eval \
#     --do_predict \
#     --evaluate_during_training \
#     --evaluate_each_epoch \
#     --do_lower_case \
#     --save_best_checkpoints \
#     --seed=42

python run_span.py \
    --version=nezha \
    --device=cuda:0 \
    --n_gpu=1 \
    --task_name=ner \
    --dataset_name=cail_ner \
    --data_dir=./data/ner-ctx0-train0.8-seed42/ \
    --train_file=train.json \
    --dev_file=dev.json \
    --test_file=dev.json \
    --model_type=nezha_span \
    --model_name_or_path=/home/louishsu/NewDisk/Garage/weights/transformers/nezha-cn-base/ \
    --max_span_length=30 \
    --width_embedding_dim=128 \
    --train_max_seq_length=512 \
    --eval_max_seq_length=512 \
    --per_gpu_train_batch_size=16 \
    --per_gpu_eval_batch_size=16 \
    --gradient_accumulation_steps=2 \
    --learning_rate=2e-5 \
    --other_learning_rate=1e-3 \
    --num_train_epochs=8.0 \
    --do_train \
    --do_eval \
    --do_predict \
    --evaluate_during_training \
    --evaluate_each_epoch \
    --do_lower_case \
    --save_best_checkpoints \
    --seed=42 \
    --fp16