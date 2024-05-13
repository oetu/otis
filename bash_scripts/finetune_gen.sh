#!/usr/bin/bash
# Pre-training

# Basic parameters
seed="0"
num_workers="4"    # number of CPUs

path="tower"       # [tower, server]
submitit="False"    # only for training on server

nodes="1"
world_size="1"      # number of GPUs
mem_per_task="69"   # memory per GPU
port="29408"

batch_size="1"
accum_iter=(1)

epochs="1000"
warmup_epochs="50"

# Callback parameters
patience="-1"
max_delta="0.00"

# Model parameters
compile="False"

model_size="baseDeep_dec160d4b"
model="otis_"$model_size"_patchX"

univariate="True"

output_projection="decoder"

# Pretraining specifications
from_scratch="False"

ignore_pos_embed_y="False"
freeze_pos_embed_y="False"
freeze_encoder="False"
ignore_decoder="False"

input_channels="1"
input_electrodes="7"
time_steps="384" # 384, 192

patch_height="1"
patch_width=(24)

separate_pos_embed_y="False"

# Loss parameters
norm_pix_loss="False"
masked_patch_loss="False"
domain_weighted_loss="False"

ncc_weight=0.0
cos_weight=0.0

# Augmentation parameters
mask_ratio=(0.125) # 0.125, 0.25

crop_lower_bnd="1.0"
crop_upper_bnd="1.0"

jitter_sigma="0.0"
rescaling_sigma="0.0"
ft_surr_phase_noise="0.0"

# Optimizer parameters
blr_array=(3e-1) # 3e-1, 1e-2
weight_decay=(0.15)

downstream_task="forecasting"

# Output path
folder="otis_gen"

# Data path
dataset="etth"

# Log specifications
save_output="False"
wandb="True"
wandb_project="OTiS_Generative_Tasks"
wandb_id=""

if [ "$path" = "tower" ]; then
    if [ "$dataset" = "ukbb" ]; then
        data_base="/home/oturgut/data/processed/UKBB"
    elif  [ "$dataset" = "mimic" ]; then
        data_base="/home/oturgut/data/processed/mimic-ecg-text"
    elif [ "$dataset" = "etth" ]; then
        data_base="/home/oturgut/data/processed/benchmarks/forecasting"
    elif [ "$dataset" = "ili" ]; then
        data_base="/home/oturgut/data/processed/benchmarks/forecasting"
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
    elif [ "$dataset" = "etth" ]; then
        data_base="/vol/aimspace/users/tuo/data/processed/benchmarks/forecasting"
    elif [ "$dataset" = "ili" ]; then
        data_base="/vol/aimspace/users/tuo/data/processed/benchmarks/forecasting"
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
elif [ "$dataset" = "etth" ]; then
    data_path=$data_base"/data_etth_all.pt"
    val_data_path=$data_base"/data_etth_all.pt"
elif [ "$dataset" = "ili" ]; then
    data_path=$data_base"/data_ili_all.pt"
    val_data_path=$data_base"/data_ili_all.pt"
elif [ "$dataset" = "ticorp" ]; then
    data_path=$data_base"/train.pt"
    val_data_path=$data_base"/val.pt"
else
    data_path=$data_base"/data_train_new.pt"
    val_data_path=$data_base"/data_val_new.pt"
fi

# EVALUATE
eval="False"
# As filename: State the checkpoint for the inference of a specific model
# or state the (final) epoch for the inference of all models up to this epoch
#resume=$checkpoint_base"/output/fin/"$folder"/id/"$subfolder"/fin_b"$(($batch_size*$accum_iter))"_blr"$blr"/checkpoint-89.pth"

for blr in "${blr_array[@]}"
do
    for acc_it in "${accum_iter[@]}"
    do
        for mr in "${mask_ratio[@]}"
        do

            subfolder="cos_weight$cos_weight/ncc_weight$ncc_weight/seed$seed/$model_size/t$time_steps/p$patch_height"x"$patch_width/wd$weight_decay/m$mr"

            output_dir=$checkpoint_base"/output/fin/"$folder"/"$subfolder"/pre_b"$(($batch_size*$acc_it*$world_size))"_blr"$blr

            # resume=$checkpoint_base"/output/pre/"$folder"/"$subfolder"/"$pre_data"/checkpoint-60-ncc-0.5985.pth"

            # OTiS
            # base
            # finetune="/home/oturgut/SiT/output/pre/otis/ticorp/ft/cos_weight0.0/ncc_weight0.1/seed0/baseDeep_dec160d4b/t1008/p1x24/wd0.15/m0.8/pre_b544_blr3e-4/checkpoint-93-ncc-0.7326.pth"
            finetune="/home/oturgut/SiT/output/pre/otis/base/dec160d4b/p1x24/pre_b1216_blr3e-5/checkpoint-99-ncc-0.8662.pth"
            # finetune="/home/oturgut/SiT/output/pre/otis/base/dec128d2b/p1x24/pre_b1792_blr3e-5/checkpoint-92-ncc-0.8690.pth"
            # finetune="/home/oturgut/SiT/output/pre/otis/base/dec128d2b/p1x24/pre_b1792_blr3e-5/checkpoint-98-ncc-0.8632.pth"

            # large
            # finetune="/home/oturgut/SiT/output/pre/otis/large/dec128d2b/p1x24/pre_896_blr3e-5/checkpoint-98-ncc-0.8688.pth"

            # huge
            # finetune="/home/oturgut/SiT/output/pre/otis/huge/dec128d2b/p1x24/pre_b1024_blr1e-5/checkpoint-96-ncc-0.8690.pth"
            # finetune="/home/oturgut/SiT/output/pre/otis/huge/dec160d4b/p1x24/pre_b2912_blr3e-6/checkpoint-99-ncc-0.8705.pth"
            # finetune="/home/oturgut/SiT/output/pre/otis/huge/dec128d2b/p1x24/pre_b1024_blr1e-5/checkpoint-99-ncc-0.8655.pth"
            
            if [ "$path" = "tower" ]; then
                cmd="python3 main_finetune_gen.py --output_projection $output_projection --downstream_task $downstream_task --seed $seed --patience $patience --crop_lower_bnd $crop_lower_bnd --crop_upper_bnd $crop_upper_bnd --max_delta $max_delta --jitter_sigma $jitter_sigma --rescaling_sigma $rescaling_sigma --ft_surr_phase_noise $ft_surr_phase_noise --input_channels $input_channels --input_electrodes $input_electrodes --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --ncc_weight $ncc_weight --cos_weight $cos_weight --model $model --batch_size $batch_size --epochs $epochs --accum_iter $acc_it --mask_ratio $mr --weight_decay $weight_decay --blr $blr --warmup_epoch $warmup_epochs --data_path $data_path --val_data_path $val_data_path --num_workers $num_workers"
            elif [ "$submitit" = "True" ]; then
                cmd="python3 submitit_finetune_gen.py --mem_per_task $mem_per_task --ngpus $world_size --nodes $nodes --output_projection $output_projection --downstream_task $downstream_task --seed $seed --patience $patience --crop_lower_bnd $crop_lower_bnd --crop_upper_bnd $crop_upper_bnd --max_delta $max_delta --jitter_sigma $jitter_sigma --rescaling_sigma $rescaling_sigma --ft_surr_phase_noise $ft_surr_phase_noise --input_channels $input_channels --input_electrodes $input_electrodes --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --ncc_weight $ncc_weight --cos_weight $cos_weight --model $model --batch_size $batch_size --epochs $epochs --accum_iter $acc_it --mask_ratio $mr --weight_decay $weight_decay --blr $blr --warmup_epoch $warmup_epochs --data_path $data_path --val_data_path $val_data_path --num_workers $num_workers"
            else
                cmd="torchrun --rdzv-endpoint=localhost:$port --nproc_per_node $world_size --nnodes $nodes --node_rank 0 main_finetune_gen.py --world_size $world_size --dist_eval --output_projection $output_projection --downstream_task $downstream_task --seed $seed --patience $patience --crop_lower_bnd $crop_lower_bnd --crop_upper_bnd $crop_upper_bnd --max_delta $max_delta --jitter_sigma $jitter_sigma --rescaling_sigma $rescaling_sigma --ft_surr_phase_noise $ft_surr_phase_noise --input_channels $input_channels --input_electrodes $input_electrodes --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --ncc_weight $ncc_weight --cos_weight $cos_weight --model $model --batch_size $batch_size --epochs $epochs --accum_iter $acc_it --mask_ratio $mr --weight_decay $weight_decay --blr $blr --warmup_epoch $warmup_epochs --data_path $data_path --val_data_path $val_data_path --num_workers $num_workers"
            fi

            if [ "$univariate" = "True" ]; then
                cmd=$cmd" --univariate"
            fi

            if [ "$ignore_pos_embed_y" = "True" ]; then
                cmd=$cmd" --ignore_pos_embed_y"
            fi

            if [ "$freeze_pos_embed_y" = "True" ]; then
                cmd=$cmd" --freeze_pos_embed_y"
            fi

            if [ "$freeze_encoder" = "True" ]; then
                cmd=$cmd" --freeze_encoder"
            fi

            if [ "$ignore_decoder" = "True" ]; then
                cmd=$cmd" --ignore_decoder"
            fi
            
            if [ "$from_scratch" = "False" ]; then
                cmd=$cmd" --finetune $finetune"
            fi

            if [ "$compile" = "True" ]; then
                cmd=$cmd" --compile"
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

            if [ "$wandb" = "True" ]; then
                cmd=$cmd" --wandb --wandb_project $wandb_project"
                if [ ! -z "$wandb_id" ]; then
                    cmd=$cmd" --wandb_id $wandb_id"
                fi
            fi

            if [ "$save_output" = "True" ]; then
                cmd=$cmd" --output_dir $output_dir"
            fi

            if [ "$eval" = "True" ]; then
                cmd=$cmd" --eval --resume $resume"
            fi

            if [ ! -z "$resume" ]; then
                cmd=$cmd" --resume $resume"
            fi

            echo $cmd && $cmd

        done
    done
done