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
python prepare_data.py \
    --data_files ./data/信息抽取_第二阶段/xxcq_mid.json \
    --context_window 0 \
    --n_splits 5 \
    --output_dir data/ \
    --seed 42
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
# avg
# {'p': 0.9142199194012666, 'r': 0.9188042430086789, 'f': 0.9165063485956138}
# 犯罪嫌疑人
# {'p': 0.9647829647829648, 'r': 0.9546191247974068, 'f': 0.959674134419552}
# 受害人
# {'p': 0.8846153846153846, 'r': 0.968013468013468, 'f': 0.9244372990353698}
# 被盗货币
# {'p': 0.8152173913043478, 'r': 0.8670520231213873, 'f': 0.8403361344537815}
# 物品价值
# {'p': 0.9780487804878049, 'r': 0.9828431372549019, 'f': 0.9804400977995109}
# 盗窃获利
# {'p': 0.8780487804878049, 'r': 0.9152542372881356, 'f': 0.896265560165975}
# 被盗物品
# {'p': 0.903485254691689, 'r': 0.8829694323144105, 'f': 0.8931095406360424}
# 作案工具
# {'p': 0.7407407407407407, 'r': 0.7874015748031497, 'f': 0.7633587786259541}
# 时间
# {'p': 0.9361702127659575, 'r': 0.9219047619047619, 'f': 0.928982725527831}
# 地点
# {'p': 0.8971830985915493, 'r': 0.885952712100139, 'f': 0.8915325402379286}
# 组织机构
# {'p': 0.8450704225352113, 'r': 0.8450704225352113, 'f': 0.8450704225352113}

for k in 0 1 2 3 4
do
python run_span.py \
    --version=rdrop0.1-fgm1.0-fold${k} \
    --data_dir=./data/ner-ctx0-5fold-seed42/ \
    --train_file=train.${k}.json \
    --dev_file=dev.${k}.json \
    --test_file=dev.${k}.json \
    --model_type=bert_span \
    --model_name_or_path=/home/louishsu/NewDisk/Garage/weights/transformers/chinese-roberta-wwm/ \
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
    --num_train_epochs=4.0 \
    --warmup_proportion=0.1 \
    --rdrop_alpha=0.1 \
    --do_fgm --fgm_epsilon=1.0 \
    --seed=42
done
# main_local
# avg
# {'p': 0.8833097595473833, 'r': 0.8901016465999024, 'f': 0.8866926971434976}
# 犯罪嫌疑人
# {'p': 0.951840490797546, 'r': 0.9602351848986539, 'f': 0.9560194099976893}
# 受害人
# {'p': 0.9128682170542636, 'r': 0.9472329472329473, 'f': 0.9297331438496764}
# 被盗货币
# {'p': 0.8006134969325154, 'r': 0.8557377049180328, 'f': 0.8272583201267829}
# 物品价值
# {'p': 0.967896502156205, 'r': 0.9665071770334929, 'f': 0.9672013406751256}
# 盗窃获利
# {'p': 0.8346007604562737, 'r': 0.9126819126819127, 'f': 0.8718967229394241}
# 被盗物品
# {'p': 0.7937117903930131, 'r': 0.78602317938073, 'f': 0.7898487745524074}
# 作案工具
# {'p': 0.7383059418457648, 'r': 0.7945578231292517, 'f': 0.765399737876802}
# 时间
# {'p': 0.9410688140556369, 'r': 0.9298372513562387, 'f': 0.9354193196288886}
# 地点
# {'p': 0.8487467588591184, 'r': 0.8376457207847597, 'f': 0.8431597023468804}
# 组织机构
# {'p': 0.8557336621454994, 'r': 0.8610421836228288, 'f': 0.8583797155225728}

for k in 0 1 2 3 4
do
python run_span.py \
    --version=nezha_rdrop0.1-fgm1.0-fold${k} \
    --data_dir=./data/ner-ctx0-5fold-seed42/ \
    --train_file=train.${k}.json \
    --dev_file=dev.${k}.json \
    --test_file=dev.${k}.json \
    --model_type=nezha_span \
    --model_name_or_path=/home/louishsu/NewDisk/Garage/weights/transformers/nezha-cn-base/ \
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
    --num_train_epochs=4.0 \
    --warmup_proportion=0.1 \
    --rdrop_alpha=0.1 \
    --do_fgm --fgm_epsilon=1.0 \
    --seed=42
done
# main_local
# avg
# {'p': 0.8898079874237377, 'r': 0.8916769813585387, 'f': 0.8907415039904079}
# 犯罪嫌疑人
# {'p': 0.9548456458301336, 'r': 0.9619371808757543, 'f': 0.9583782950516417}
# 受害人
# {'p': 0.9174141664089082, 'r': 0.9543114543114544, 'f': 0.9354991326289229}
# 被盗货币
# {'p': 0.8018575851393189, 'r': 0.8491803278688524, 'f': 0.8248407643312102}
# 物品价值
# {'p': 0.9693192713326941, 'r': 0.9674641148325359, 'f': 0.9683908045977011}
# 盗窃获利
# {'p': 0.8557504873294347, 'r': 0.9126819126819127, 'f': 0.8832997987927566}
# 被盗物品
# {'p': 0.8040147913365029, 'r': 0.7898287493513233, 'f': 0.7968586387434555}
# 作案工具
# {'p': 0.7706666666666667, 'r': 0.7863945578231293, 'f': 0.7784511784511785}
# 时间
# {'p': 0.9394717534849596, 'r': 0.9262206148282097, 'f': 0.9327991258422873}
# 地点
# {'p': 0.8587241479755316, 'r': 0.8382143872618709, 'f': 0.8483453237410072}
# 组织机构
# {'p': 0.8555691554467564, 'r': 0.8672456575682382, 'f': 0.8613678373382625}

# 去掉rdrop
for k in 0 1 2 3 4
do
python run_span.py \
    --version=nezha-fgm1.0-fold${k} \
    --data_dir=./data/ner-ctx0-5fold-seed42/ \
    --train_file=train.${k}.json \
    --dev_file=dev.${k}.json \
    --test_file=dev.${k}.json \
    --model_type=nezha_span \
    --model_name_or_path=/home/louishsu/NewDisk/Garage/weights/transformers/nezha-cn-base/ \
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
    --num_train_epochs=4.0 \
    --warmup_proportion=0.1 \
    --do_fgm --fgm_epsilon=1.0 \
    --seed=42
done
# main_local
# avg
# {'p': 0.8900258630383447, 'r': 0.8906267581861146, 'f': 0.890326209223847}
# 犯罪嫌疑人
# {'p': 0.9531777709548664, 'r': 0.9606993656196813, 'f': 0.9569237882407337}
# 受害人
# {'p': 0.9161769254562326, 'r': 0.953024453024453, 'f': 0.9342375019712978}
# 被盗货币
# {'p': 0.8041666666666667, 'r': 0.8437158469945355, 'f': 0.8234666666666667}
# 物品价值
# {'p': 0.9702352376380221, 'r': 0.9669856459330144, 'f': 0.9686077162712677}
# 盗窃获利
# {'p': 0.8529980657640233, 'r': 0.9168399168399168, 'f': 0.8837675350701403}
# 被盗物品
# {'p': 0.8070051300194587, 'r': 0.7891368275384881, 'f': 0.7979709637921986}
# 作案工具
# {'p': 0.7707774798927614, 'r': 0.782312925170068, 'f': 0.7765023632680621}
# 时间
# {'p': 0.9405504587155963, 'r': 0.9269439421338156, 'f': 0.9336976320582878}
# 地点
# {'p': 0.8563938246431693, 'r': 0.8359397213534262, 'f': 0.8460431654676259}
# 组织机构
# {'p': 0.8588957055214724, 'r': 0.8684863523573201, 'f': 0.8636644046884641}

# focal
for k in 0 1 2 3 4
do
python run_span.py \
    --version=nezha-rdrop0.1-fgm1.0-focalg2.0a0.25-fold${k} \
    --data_dir=./data/ner-ctx0-5fold-seed42/ \
    --train_file=train.${k}.json \
    --dev_file=dev.${k}.json \
    --test_file=dev.${k}.json \
    --model_type=nezha_span \
    --model_name_or_path=/home/louishsu/NewDisk/Garage/weights/transformers/nezha-cn-base/ \
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
    --num_train_epochs=4.0 \
    --warmup_proportion=0.1 \
    --rdrop_alpha=0.1 \
    --do_fgm --fgm_epsilon=1.0 \
    --loss_type=focal --focal_gamma=2.0 --focal_alpha=0.25 \
    --seed=42
done
# main_local
# avg
# {'p': 0.8764269298873526, 'r': 0.8725479164322418, 'f': 0.8744831215698068}
# 犯罪嫌疑人
# {'p': 0.9334845049130763, 'r': 0.9554386507813709, 'f': 0.9443339960238568}
# 受害人
# {'p': 0.9038400507775309, 'r': 0.9163449163449163, 'f': 0.9100495286787026}
# 被盗货币
# {'p': 0.8149732620320855, 'r': 0.8327868852459016, 'f': 0.8237837837837838}
# 物品价值
# {'p': 0.9555765595463138, 'r': 0.9674641148325359, 'f': 0.9614835948644794}
# 盗窃获利
# {'p': 0.8493150684931506, 'r': 0.9022869022869023, 'f': 0.875}
# 被盗物品
# {'p': 0.7896645512239348, 'r': 0.7533298737242692, 'f': 0.7710694050991501}
# 作案工具
# {'p': 0.7427385892116183, 'r': 0.7306122448979592, 'f': 0.7366255144032923}
# 时间
# {'p': 0.9340900768949103, 'r': 0.9226039783001808, 'f': 0.9283114992721979}
# 地点
# {'p': 0.8347725964306275, 'r': 0.8245663918112027, 'f': 0.8296381061364613}
# 组织机构
# {'p': 0.8795336787564767, 'r': 0.8424317617866005, 'f': 0.8605830164765527}

# context-aware，仅增强分类性能较差的“受害人、犯罪嫌疑人”
for k in 0 1 2 3 4
do
python run_span.py \
    --version=nezha-rdrop0.1-fgm1.0-aug_ctx0.15-fold${k} \
    --data_dir=./data/ner-ctx0-5fold-seed42/ \
    --train_file=train.${k}.json \
    --dev_file=dev.${k}.json \
    --test_file=dev.${k}.json \
    --model_type=nezha_span \
    --model_name_or_path=/home/louishsu/NewDisk/Garage/weights/transformers/nezha-cn-base/ \
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
    --num_train_epochs=4.0 \
    --warmup_proportion=0.1 \
    --do_fgm --fgm_epsilon=1.0 \
    --augment_context_aware_p=0.15 \
    --seed=42
done
# main_local
# avg
# {'p': 0.8945034116755117, 'r': 0.885075578560444, 'f': 0.8897645217850342}
# 犯罪嫌疑人
# {'p': 0.9578052550231839, 'r': 0.9588426427355717, 'f': 0.9583236681357767}
# 受害人
# {'p': 0.9202144433932513, 'r': 0.9388674388674388, 'f': 0.929447364229973}
# 被盗货币
# {'p': 0.8020304568527918, 'r': 0.8633879781420765, 'f': 0.831578947368421}
# 物品价值
# {'p': 0.9693192713326941, 'r': 0.9674641148325359, 'f': 0.9683908045977011}
# 盗窃获利
# {'p': 0.8552123552123552, 'r': 0.920997920997921, 'f': 0.8868868868868868}
# 被盗物品
# {'p': 0.8162181951308805, 'r': 0.771319840857983, 'f': 0.7931341159729633}
# 作案工具
# {'p': 0.7618421052631579, 'r': 0.7877551020408163, 'f': 0.774581939799331}
# 时间
# {'p': 0.9468796433878157, 'r': 0.9218806509945751, 'f': 0.9342129375114532}
# 地点
# {'p': 0.863689776733255, 'r': 0.8359397213534262, 'f': 0.8495882097962723}
# 组织机构
# {'p': 0.8423586040914561, 'r': 0.8684863523573201, 'f': 0.855222968845449}

# LSR
for k in 0 1 2 3 4
do
python run_span.py \
    --version=nezha-fgm1.0-lsr0.1-fold${k} \
    --data_dir=./data/ner-ctx0-5fold-seed42/ \
    --train_file=train.${k}.json \
    --dev_file=dev.${k}.json \
    --test_file=dev.${k}.json \
    --model_type=nezha_span \
    --model_name_or_path=/home/louishsu/NewDisk/Garage/weights/transformers/nezha-cn-base/ \
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
# main_local
# avg
# {'p': 0.8992657967816792, 'r': 0.8866509133190803, 'f': 0.8929138022210471}
# 犯罪嫌疑人
# {'p': 0.9609907120743034, 'r': 0.9605446387126721, 'f': 0.9607676236168073}
# 受害人
# {'p': 0.9242566510172144, 'r': 0.9501287001287001, 'f': 0.9370141202601936}
# 被盗货币
# {'p': 0.8159574468085107, 'r': 0.8382513661202186, 'f': 0.8269541778975741}
# 物品价值
# {'p': 0.9697696737044146, 'r': 0.9669856459330144, 'f': 0.9683756588404409}
# 盗窃获利
# {'p': 0.8627450980392157, 'r': 0.9147609147609148, 'f': 0.8879919273461151}
# 被盗物品
# {'p': 0.8217407137654771, 'r': 0.7806607853312576, 'f': 0.8006741772376476}
# 作案工具
# {'p': 0.801994301994302, 'r': 0.7659863945578231, 'f': 0.7835768963117605}
# 时间
# {'p': 0.9488699518340126, 'r': 0.9262206148282097, 'f': 0.9374084919472914}
# 地点
# {'p': 0.861102919492775, 'r': 0.8302530565823145, 'f': 0.8453966415749856}
# 组织机构
# {'p': 0.8513513513513513, 'r': 0.8598014888337469, 'f': 0.8555555555555555}

# Further-pretrain LSR
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
# main_local
# avg
# {'p': 0.9032722314800787, 'r': 0.8945650950827051, 'f': 0.898897578441534}
# 犯罪嫌疑人
# {'p': 0.9654852190063458, 'r': 0.965186445922946, 'f': 0.9653358093469513}
# 受害人
# {'p': 0.9271829682196853, 'r': 0.9668597168597168, 'f': 0.9466057646873522}
# 被盗货币
# {'p': 0.8280590717299579, 'r': 0.8579234972677595, 'f': 0.8427267847557702}
# 物品价值
# {'p': 0.9745069745069745, 'r': 0.969377990430622, 'f': 0.9719357159990405}
# 盗窃获利
# {'p': 0.8812877263581489, 'r': 0.9106029106029107, 'f': 0.8957055214723928}
# 被盗物品
# {'p': 0.8194369732831271, 'r': 0.7905206711641585, 'f': 0.8047191406937841}
# 作案工具
# {'p': 0.7972972972972973, 'r': 0.8027210884353742, 'f': 0.7999999999999999}
# 时间
# {'p': 0.950575994054255, 'r': 0.9251356238698011, 'f': 0.937683284457478}
# 地点
# {'p': 0.8714835652946402, 'r': 0.836792721069093, 'f': 0.8537859007832898}
# 组织机构
# {'p': 0.8789407313997478, 'r': 0.8647642679900744, 'f': 0.8717948717948718}

# TODO: 全部数据

# TODO: EMA
for k in 0 1 2 3 4
do
python run_span.py \
    --version=nezha-fgm1.0-ema0.999-fold${k} \
    --data_dir=./data/ner-ctx0-5fold-seed42/ \
    --train_file=train.${k}.json \
    --dev_file=dev.${k}.json \
    --test_file=dev.${k}.json \
    --model_type=nezha_span \
    --model_name_or_path=/home/louishsu/NewDisk/Garage/weights/transformers/nezha-cn-base/ \
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
    --num_train_epochs=4.0 \
    --warmup_proportion=0.1 \
    --do_fgm --fgm_epsilon=1.0 \
    --do_ema \
    --seed=42
done

# TODO: albert
# TODO: distill
# TODO: further pretrain，加入别的赛道数据
# TODO: 伪标签，别的赛道数据作为无标签数据
# TODO: 往年数据
# TODO: TTA

# <<< 第二阶段 <<<

# ==================================================================================================================

# for k in 0 1 2 3 4
# do
# python run_span.py \
#     --version=legal_electra_base-fold${k} \
#     --data_dir=./data/ner-ctx0-5fold-seed42/ \
#     --train_file=train.${k}.json \
#     --dev_file=dev.${k}.json \
#     --test_file=dev.${k}.json \
#     --model_type=bert_span \
#     --model_name_or_path=/home/louishsu/NewDisk/Garage/weights/transformers/hfl_chinese-legal-electra-base-discriminator/ \
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
# python evaluate.py \
#     ./data/ner-ctx0-5fold-seed42/dev.gt.${k}.json \
#     output/ner-cail_ner-bert_span-legal_electra_base-fold${k}-42/test_prediction.json
# done

# for k in 0 1 2 3 4
# do
# python run_span.py \
#     --version=nezha-rdrop0.1-fgm1.0-fp16-fold${k} \
#     --data_dir=./data/ner-ctx0-5fold-seed42/ \
#     --train_file=train.${k}.json \
#     --dev_file=dev.${k}.json \
#     --test_file=dev.${k}.json \
#     --model_type=nezha_span \
#     --model_name_or_path=/home/louishsu/NewDisk/Garage/weights/transformers/nezha-cn-base/ \
#     --do_train \
#     --overwrite_output_dir \
#     --evaluate_during_training \
#     --evaluate_each_epoch \
#     --save_best_checkpoints \
#     --max_span_length=40 \
#     --width_embedding_dim=128 \
#     --train_max_seq_length=512 \
#     --eval_max_seq_length=512 \
#     --do_lower_case \
#     --per_gpu_train_batch_size=6 \
#     --per_gpu_eval_batch_size=12 \
#     --gradient_accumulation_steps=2 \
#     --learning_rate=5e-5 \
#     --other_learning_rate=1e-3 \
#     --num_train_epochs=4.0 \
#     --warmup_proportion=0.1 \
#     --rdrop_alpha=0.1 \
#     --do_fgm --fgm_epsilon=1.0 \
#     --seed=42 \
#     --fp16
# done

# for k in 0 1 2 3 4
# do
# python run_span.py \
#     --version=nezha-rdrop0.1-vat0.1-fgm1.0-fp16-fold${k} \
#     --data_dir=./data/ner-ctx0-5fold-seed42/ \
#     --train_file=train.${k}.json \
#     --dev_file=dev.${k}.json \
#     --test_file=dev.${k}.json \
#     --model_type=nezha_span \
#     --model_name_or_path=/home/louishsu/NewDisk/Garage/weights/transformers/nezha-cn-base/ \
#     --do_train \
#     --overwrite_output_dir \
#     --evaluate_during_training \
#     --evaluate_each_epoch \
#     --save_best_checkpoints \
#     --max_span_length=40 \
#     --width_embedding_dim=128 \
#     --train_max_seq_length=512 \
#     --eval_max_seq_length=512 \
#     --do_lower_case \
#     --per_gpu_train_batch_size=6 \
#     --per_gpu_eval_batch_size=12 \
#     --gradient_accumulation_steps=2 \
#     --learning_rate=5e-5 \
#     --other_learning_rate=1e-3 \
#     --num_train_epochs=4.0 \
#     --warmup_proportion=0.1 \
#     --rdrop_alpha=0.1 \
#     --do_fgm --fgm_epsilon=1.0 \
#     --vat_alpha=0.1 \
#     --seed=42 \
#     --fp16
# done