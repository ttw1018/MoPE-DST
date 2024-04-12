#!/bin/bash

backbone="chatglm"
total_step=100000
save_step=5000

dataset=multiwoz2.1

train_batch_size=3
evaluate_batch_size=24
none_rate=1

exclude_domains=(attraction hotel restaurant taxi train)


feature="transformer"
for exclude_domain in ${exclude_domains[*]}; do
  for n_clusters in 1 2; do
    result_csv_dir="result/zero-shot"
    train_id=${n_clusters}-cluster-V3
    accelerate launch \
      train.py \
      --dataset $dataset \
      --backbone $backbone \
      --train_batch_size ${train_batch_size} \
      --evaluate_batch_size ${evaluate_batch_size} \
      --use_single_state \
      --mppt \
      --lr 1e-2 \
      --save_model \
      --plot_loss \
      --gradient_accumulation_steps 1 \
      --result_csv_dir $result_csv_dir \
      --train_id $train_id \
      --use_lower \
      --n_clusters ${n_clusters} \
      --total_step ${total_step} \
      --save_step ${save_step} \
      --cluster_feature $feature \
      --zero_shot \
      --exclude_domain ${exclude_domain} \
      --none_rate ${none_rate} \
      --use_all_state \
      --chatglm_path /home/hadoop-mining/dolphinfs_hdd_hadoop-mining/tianwen/ptm/glm \
      --stop_metrics slot_acc

    python evaluate.py \
      --mppt \
      --dataset $dataset \
      --backbone $backbone \
      --evaluate_batch_size ${evaluate_batch_size} \
      --use_single_state \
      --save_prediction \
      --train_id ${train_id} \
      --checkpoint save/${dataset}/${exclude_domain}/${train_id}/${feature}/best \
      --use_lower \
      --n_clusters ${n_clusters} \
      --result_csv_dir $result_csv_dir \
      --zero_shot \
      --cluster_feature $feature \
      --exclude_domain ${exclude_domain} \
      --none_rate ${none_rate} \
      --use_all_state \
      --chatglm_path /home/hadoop-mining/dolphinfs_hdd_hadoop-mining/tianwen/ptm/glm 
  done
done
