#!/usr/bin/bash
# Pre-training

# Basic parameters
seed="0"
num_workers="32"    # number of CPUs

path="server"       # [tower, server]
submitit="False"     # only for training on server

nodes="1"
world_size="8"      # number of GPUs
mem_per_task="200"   # memory per GPU
port="29443"

batch_size="512"
accum_iter=(1)

epochs="100"
warmup_epochs="10"

# Callback parameters
patience="-1"
max_delta="0.00"

# Model parameters
compile="False"

model_size="baseDeep_dec128d2b"
model="otis_"$model_size"_patchX"

input_channels="1"
time_steps="1008"

patch_height="1"
patch_width=(24)

separate_pos_embed_y="False"

norm_pix_loss="False"
masked_patch_loss="False"
domain_weighted_loss="True"

ncc_weight=0.1
cos_weight=0.0

# Augmentation parameters
mask_ratio=(0.75)
include_forecasting_mask="True"

crop_lower_bnd="0.5"
crop_upper_bnd="1.0"

jitter_sigma="0.25"
rescaling_sigma="0.5"
ft_surr_phase_noise="0.1"

# Optimizer parameters
blr_array=(1e-4)
weight_decay=(0.15)

# Data path
dataset="ticorp_light"

if [ "$path" = "tower" ]; then
    if [ "$dataset" = "ukbb" ]; then
        data_base="/home/oturgut/data/processed/ukbb"
    elif  [ "$dataset" = "mimic" ]; then
        data_base="/home/oturgut/data/processed/mimic-ecg-text"
    elif  [ "$dataset" = "ticorp" ]; then
        data_base="/home/oturgut/data/processed/TiCorp"
    else 
        data_base="/home/oturgut/data/processed/TiCorp"
    fi
    checkpoint_base="/home/oturgut/SiT"
else
    if [ "$dataset" = "ukbb" ]; then
        data_base="/vol/aimspace/projects/ukbb/data/cardiac/cardiac_segmentations/projects/ecg"
    elif [ "$dataset" = "mimic" ]; then
        data_base="/vol/aimspace/projects/physionet/mimic/processed/mimic-ecg-text"
    elif  [ "$dataset" = "ticorp" ]; then
        data_base="/vol/aimspace/users/tuo/data/processed/TiCorp"
    else
        data_base="/vol/aimspace/users/tuo/data/processed/TiCorp"
    fi
    checkpoint_base="/vol/aimspace/users/tuo/SiT"
fi

# Dataset parameters
if [ "$dataset" = "ukbb" ]; then
    data_path=$data_base"/otis/ecgs_train_ecg_imaging_float32.pt"
    val_data_path=$data_base"/otis/ecgs_val_ecg_imaging_float32.pt"
elif [ "$dataset" = "mimic" ]; then
    data_path=$data_base"/ecgs_train_300k.pt"
    val_data_path=$data_base"/ecgs_val_10k.pt"
elif [ "$dataset" = "ticorp" ]; then
    data_path=$data_base"/train.pt"
    val_data_path=$data_base"/val.pt"
else
    data_path=$data_base"/train_light.pt"
    val_data_path=$data_base"/val.pt"
fi

# Online evaluation
input_electrodes="12"
online_evaluation="True"
online_evaluation_task="classification"
lower_bnd="0"
upper_bnd="1"
online_num_classes=2

if [ "$path" = "tower" ]; then # ukbb data for online eval
    online_data_base="/home/oturgut/data/processed/ukbb"
else
    online_data_base="/vol/aimspace/projects/ukbb/data/cardiac/cardiac_segmentations/projects/ecg"
fi

target="CAD"
data_path_online=$online_data_base"/otis/ecgs_train_"$target"_all_balanced_float32.pt"
labels_path_online=$online_data_base"/labelsOneHot/labels_train_"$target"_all_balanced.pt"

val_data_path_online=$online_data_base"/otis/ecgs_val_ecg_imaging_float32.pt"
val_labels_path_online=$online_data_base"/labelsOneHot/labels_val_"$target"_all.pt"

# Log specifications
save_output="True"
wandb="True"
wandb_project="OTiS_Pretraining"
wandb_id=""

for blr in "${blr_array[@]}"
do
    for acc_it in "${accum_iter[@]}"
    do
        for mr in "${mask_ratio[@]}"
        do

            folder="otis_refactored/fm0.1"
            subfolder="cos_weight$cos_weight/ncc_weight$ncc_weight/seed$seed/$model_size/t$time_steps/p$patch_height"x"$patch_width/wd$weight_decay/m$mr"

            output_dir=$checkpoint_base"/output/pre/"$folder"/"$subfolder"/pre_b"$(($batch_size*$acc_it*$world_size))"_blr"$blr

            # resume=$checkpoint_base"/output/pre/"$folder"/"$subfolder"/"$pre_data"/checkpoint-60-ncc-0.5985.pth"
        
            if [ "$path" = "tower" ]; then
                cmd="python3 main_pretrain.py --seed $seed --patience $patience --crop_lower_bnd $crop_lower_bnd --crop_upper_bnd $crop_upper_bnd --max_delta $max_delta --jitter_sigma $jitter_sigma --rescaling_sigma $rescaling_sigma --ft_surr_phase_noise $ft_surr_phase_noise --input_channels $input_channels --input_electrodes $input_electrodes --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --ncc_weight $ncc_weight --cos_weight $cos_weight --model $model --batch_size $batch_size --epochs $epochs --accum_iter $acc_it --mask_ratio $mr --weight_decay $weight_decay --blr $blr --warmup_epoch $warmup_epochs --data_path $data_path --val_data_path $val_data_path --num_workers $num_workers"
            elif [ "$submitit" = "True" ]; then
                cmd="python3 submitit_pretrain.py --mem_per_task $mem_per_task --ngpus $world_size --nodes $nodes --seed $seed --patience $patience --crop_lower_bnd $crop_lower_bnd --crop_upper_bnd $crop_upper_bnd --max_delta $max_delta --jitter_sigma $jitter_sigma --rescaling_sigma $rescaling_sigma --ft_surr_phase_noise $ft_surr_phase_noise --input_channels $input_channels --input_electrodes $input_electrodes --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --ncc_weight $ncc_weight --cos_weight $cos_weight --model $model --batch_size $batch_size --epochs $epochs --accum_iter $acc_it --mask_ratio $mr --weight_decay $weight_decay --blr $blr --warmup_epoch $warmup_epochs --data_path $data_path --val_data_path $val_data_path --num_workers $num_workers"
            else
                cmd="torchrun --rdzv-endpoint=localhost:$port --nproc_per_node $world_size --nnodes $nodes --node_rank 0 main_pretrain.py --world_size $world_size --dist_eval --seed $seed --patience $patience --crop_lower_bnd $crop_lower_bnd --crop_upper_bnd $crop_upper_bnd --max_delta $max_delta --jitter_sigma $jitter_sigma --rescaling_sigma $rescaling_sigma --ft_surr_phase_noise $ft_surr_phase_noise --input_channels $input_channels --input_electrodes $input_electrodes --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --ncc_weight $ncc_weight --cos_weight $cos_weight --model $model --batch_size $batch_size --epochs $epochs --accum_iter $acc_it --mask_ratio $mr --weight_decay $weight_decay --blr $blr --warmup_epoch $warmup_epochs --data_path $data_path --val_data_path $val_data_path --num_workers $num_workers"
            fi

            if [ "$compile" = "True" ]; then
                cmd=$cmd" --compile"
            fi

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

            if [ "$separate_pos_embed_y" = "True" ]; then
                cmd=$cmd" --separate_pos_embed_y"
            fi

            if [ "$masked_patch_loss" = "True" ]; then
                cmd=$cmd" --masked_patch_loss"
            fi

            if [ "$domain_weighted_loss" = "True" ]; then
                cmd=$cmd" --domain_weighted_loss"
            fi

            if [ "$include_forecasting_mask" = "True" ]; then
                cmd=$cmd" --include_forecasting_mask"
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