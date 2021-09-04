# python run_span.py \
#     --version=baseline \
#     --data_dir=./data/ner-ctx0-5fold-seed42/ \
#     --train_file=train.json \
#     --dev_file=dev.json \
#     --test_file=test.json \
#     --model_type=bert_span \
#     --model_name_or_path=/home/louishsu/NewDisk/Garage/weights/transformers/chinese-roberta-wwm/ \
#     --do_train \
#     --do_eval \
#     --do_predict \
#     --overwrite_output_dir \
#     --evaluate_during_training \
#     --evaluate_each_epoch \
#     --save_best_checkpoints \
#     --max_span_length=50 \
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
    --num_train_epochs=5.0 \
    --warmup_proportion=0.1 \
    --seed=42
python evaluate.py \
    ./data/ner-ctx0-5fold-seed42/dev.gt.${k}.json \
    output/ner-cail_ner-bert_span-baseline-fold${k}-42/test_prediction.json
done