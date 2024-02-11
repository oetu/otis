#!/usr/bin/bash
# Pre-training

# Basic parameters
seed="0"
batch_size="256"
accum_iter=(1)

epochs="200"
warmup_epochs="20"

# Callback parameters
patience="-1"
max_delta="0.00"

# Model parameters
input_channels="1"
input_electrodes="10"
time_steps="3000"
model_size="tinyUp"
model="mae_vit_"$model_size"_patchX"

patch_height="1"
patch_width=(100)

norm_pix_loss="False"

ncc_weight=0.1

# Augmentation parameters
mask_ratio=(0.8)

jitter_sigma="0.25"
rescaling_sigma="0.5"
ft_surr_phase_noise="0.1"

# Optimizer parameters
blr_array=(1e-5)
weight_decay=(0.15)

# Data path
path="tower"
dataset="tuh/250Hz"
if [ "$path" = "tower" ]; then
    data_base="/home/oturgut/data/processed/"$dataset
    checkpoint_base="/home/oturgut/mae"
else
    data_base="/vol/aimspace/projects/physionet/mimic/processed/mimic-ecg-text"
    checkpoint_base="/vol/aimspace/users/tuo/mae"
fi

# Dataset parameters
data_path=$data_base"/data_train.pt"
val_data_path=$data_base"/data_val.pt"

num_workers="24"

# Online evaluation
online_evaluation="True"
online_evaluation_task="regression"
lower_bnd="0"
upper_bnd="1"
online_num_classes=1

online_data_base="/home/oturgut/data/processed/lemon/full"
data_path_online=$online_data_base"/data_train.pt"
labels_path_online=$online_data_base"/labels_train_stdNormed.pt"
# labels_mask_path_online=""
val_data_path_online=$online_data_base"/data_val.pt"
val_labels_path_online=$online_data_base"/labels_val_stdNormed.pt"
# val_labels_mask_path_online=""

# Log specifications
save_output="True"
wandb="True"
wandb_project="MAE_EEG_Pre"
wandb_id=""

for blr in "${blr_array[@]}"
do
    for acc_it in "${accum_iter[@]}"
    do
        for mr in "${mask_ratio[@]}"
        do

            pre_data="pre_b"$(($batch_size*$acc_it))"_blr"$blr

            folder=$dataset"/eeg/10ch"
            subfolder="ncc_weight$ncc_weight/seed$seed/$model_size/t$time_steps/p$patch_height"x"$patch_width/wd$weight_decay/m$mr"

            output_dir=$checkpoint_base"/output/pre/"$folder"/"$subfolder"/"$pre_data
                
            cmd="python3 main_pretrain.py --seed $seed --patience $patience --max_delta $max_delta --jitter_sigma $jitter_sigma --rescaling_sigma $rescaling_sigma --ft_surr_phase_noise $ft_surr_phase_noise --input_channels $input_channels --input_electrodes $input_electrodes --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --ncc_weight $ncc_weight --model $model --batch_size $batch_size --epochs $epochs --accum_iter $acc_it --mask_ratio $mr --weight_decay $weight_decay --blr $blr --warmup_epoch $warmup_epochs --data_path $data_path --val_data_path $val_data_path --num_workers $num_workers"

            if [ "$online_evaluation" = "True" ]; then
                cmd=$cmd" --online_evaluation --online_evaluation_task $online_evaluation_task --data_path_online $data_path_online --labels_path_online $labels_path_online --val_data_path_online $val_data_path_online --val_labels_path_online $val_labels_path_online"
            
                if [ "$online_evaluation_task" = "regression" ]; then
                    cmd=$cmd" --lower_bnd $lower_bnd --upper_bnd $upper_bnd"
                fi
            
                if [ ! -z "$labels_mask_path_online" ]; then
                    cmd=$cmd" --labels_mask_path_online $labels_mask_path_online"
                fi

                if [ ! -z "$val_labels_mask_path_online" ]; then
                    cmd=$cmd" --val_labels_mask_path_online $val_labels_mask_path_online"
                fi
            fi

            if [ "$norm_pix_loss" = "True" ]; then
                cmd=$cmd" --norm_pix_loss"
            fi

            if [ "$wandb" = "True" ]; then
                cmd=$cmd" --wandb --wandb_project $wandb_project"
                if [ ! -z "$wandb_id" ]; then
                    cmd=$cmd" --wandb_id $wandb_id"
                fi
            fi

            if [ "$save_output" = "True" ]; then
                cmd=$cmd" --output_dir $output_dir"
            fi

            if [ ! -z "$resume" ]; then
                cmd=$cmd" --resume $resume"
            fi

            echo $cmd && $cmd

        done
    done
done