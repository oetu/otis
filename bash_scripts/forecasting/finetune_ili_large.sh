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
port="29420"

batch_size=(1)
accum_iter=(1)

epochs="250"
warmup_epochs="25"

# Callback parameters
patience="-1"
max_delta="0.00"

eval_criterion="mse"

# Model parameters
compile="False"

model_size="largeDeep"
model="otis_"$model_size"_dec160d4b_patchX"

univariate="False"

output_projection="decoder"

# Pretraining specifications
from_scratch="False"

ignore_pos_embed_y="False"
freeze_pos_embed_y="False"
freeze_encoder="False"
ignore_decoder="False"

patch_height="1"
patch_width=(24)

separate_pos_embed_y="False"

# Loss parameters
norm_pix_loss="False"
masked_patch_loss="False"
domain_weighted_loss="False"

ncc_weight=(0.0)
cos_weight=0.0

# Augmentation parameters
mask_ratio=(0.25) # 0.125, 0.25

crop_lower_bnd="1.0"
crop_upper_bnd="1.0"

jitter_sigma="0.0"
rescaling_sigma="0.0"
ft_surr_phase_noise="0.0"

# Optimizer parameters
blr_array=(3e-1)
weight_decay=(0.15)

downstream_task="forecasting"

# Output path
# folder="etth1"
# input_channels="1"
# input_variates="7"
# time_steps="432" # 384, 192

# folder="ettm1"
# input_channels="1"
# input_variates="7"
# time_steps="432" # 384, 192

# folder="etth2"
# input_channels="1"
# input_variates="7"
# time_steps="432" # 384, 192

# folder="ettm2"
# input_channels="1"
# input_variates="7"
# time_steps="432" # 384, 192

# folder="weather"
# input_channels="1"
# input_variates="21"
# time_steps="432" # 384, 192

# folder="electricity"
# input_channels="1"
# input_variates="1"
# time_steps="432" # 384, 192

folder="ili"
input_channels="1"
input_variates="7"
time_steps="96" # 384, 192

# folder="traffic"
# input_channels="1"
# input_variates="832"
# time_steps="432" # 384, 192


# Log specifications
save_output="True"
wandb="True"
wandb_entity="oturgut"
wandb_project="OTiS_Forecasting"
wandb_id=""

if [ "$path" = "tower" ]; then
    data_base="/home/oturgut/data/processed/benchmarks/forecasting/"$folder
    checkpoint_base="/home/oturgut/otis"
else
    data_base="/vol/aimspace/users/tuo/data/processed/benchmarks/forecasting/"$folder
    checkpoint_base="/vol/aimspace/users/tuo/otis"
fi

# Dataset parameters
# Training
data_path=$data_base"/train_correct_norm.pt"

# Validation
val_data_path=$data_base"/val_correct_norm.pt"

# Test
test="True"
test_data_path=$data_base"/test_correct_norm.pt"

# EVALUATE
# As filename: State the checkpoint for the inference of a specific model
# or state the (final) epoch for the inference of all models up to this epoch
# eval_ckpt="checkpoint-253-avg-100.0000.pth"
# blr=(3e-3)
# val_data_path=$data_base"/test.pt"
# val_labels_path=$data_base"/test_labels.pt"


for sd in "${seed[@]}"
do
    for bs in "${batch_size[@]}"
    do

        for blr in "${blr_array[@]}"
        do
            for acc_it in "${accum_iter[@]}"
            do
                for mr in "${mask_ratio[@]}"
                do

                    for wd in "${weight_decay[@]}"
                    do
                        for ncc in "${ncc_weight[@]}"
                        do

                            subfolder="cos_weight$cos_weight/ncc_weight$ncc/seed$sd/$model_size/t$time_steps/p$patch_height"x"$patch_width/wd$wd/m$mr"

                            if [ "$univariate" = "True" ]; then
                                subfolder="univariate/"$subfolder
                            else
                                subfolder="multivariate/"$subfolder
                            fi

                            if [ "$model_size" = "baseDeep" ]; then
                                if [ "$path" = "tower" ]; then
                                    # finetune="/home/oturgut/otis/output/pre/otis/base/dec160d4b/p1x24/pre_b2624_blr3e-5/checkpoint-99-ncc-0.8685.pth"
                                    finetune="/home/oturgut/otis/output/pre/otis/ticorp/multivariate/domain_specific/cos_weight0.0/ncc_weight0.1/seed0/baseDeep_dec160d4b/t1008/p1x24/wd0.1/m0.75/pre_b3744_blr3e-5/checkpoint-197-ncc-0.8818.pth"
                                else
                                    finetune="/vol/aimspace/users/tuo/otis/output/pre/otis/ticorp/cos_weight0.0/ncc_weight0.1/seed0/baseDeep_dec160d4b/t1008/p1x24/wd0.15/m0.75/pre_b2624_blr3e-5/checkpoint-99-ncc-0.8685.pth"
                                fi
                            elif [ "$model_size" = "largeDeep" ]; then
                                if [ "$path" = "tower" ]; then
                                    # finetune="/home/oturgut/otis/output/pre/otis/large/dec160d4b/p1x24/pre_b768_blr3e-5/checkpoint-96-ncc-0.8667.pth"
                                    finetune="/home/oturgut/otis/output/pre/otis/ticorp/multivariate/domain_specific/cos_weight0.0/ncc_weight0.1/seed0/largeDeep_dec160d4b/t1008/p1x24/wd0.15/m0.75/pre_b3680_blr1e-5/checkpoint-188-ncc-0.8919.pth"
                                else
                                    finetune="/vol/aimspace/users/tuo/otis/output/pre/otis/ticorp/cos_weight0.0/ncc_weight0.1/seed0/largeDeep_dec160d4b/t1008/p1x24/wd0.15/m0.75/pre_b768_blr3e-5/checkpoint-96-ncc-0.8667.pth"
                                fi
                            else
                                # huge
                                if [ "$path" = "tower" ]; then
                                    # finetune="/home/oturgut/otis/output/pre/otis/huge/dec160d4b/p1x24/pre_b1680_blr1e-5/checkpoint-98-ncc-0.8661.pth"
                                    finetune="/home/oturgut/otis/output/pre/otis/ticorp/multivariate/domain_specific/cos_weight0.0/ncc_weight0.1/seed0/hugeDeep_dec160d4b/t1008/p1x24/wd0.05/m0.75/pre_b4320_blr3e-6/checkpoint-196-ncc-0.8827.pth"
                                else
                                    finetune="/vol/aimspace/users/tuo/otis/output/pre/otis/ticorp/cos_weight0.0/ncc_weight0.1/seed0/hugeDeep_dec160d4b/t1008/p1x24/wd0.15/m0.75/pre_b1680_blr1e-5/checkpoint-98-ncc-0.8661.pth"
                                fi
                            fi

                            output_dir=$checkpoint_base"/output/gen/"$folder"/"$subfolder"/pre_b"$(($bs*$acc_it*$world_size))"_blr"$blr

                            if [ "$path" = "tower" ]; then
                                cmd="python3 main_finetune_gen.py --output_projection $output_projection --downstream_task $downstream_task --seed $sd --patience $patience --crop_lower_bnd $crop_lower_bnd --crop_upper_bnd $crop_upper_bnd --max_delta $max_delta --jitter_sigma $jitter_sigma --rescaling_sigma $rescaling_sigma --ft_surr_phase_noise $ft_surr_phase_noise --input_channels $input_channels --input_variates $input_variates --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --ncc_weight $ncc --cos_weight $cos_weight --model $model --batch_size $bs --epochs $epochs --accum_iter $acc_it --mask_ratio $mr --weight_decay $wd --blr $blr --warmup_epochs $warmup_epochs --data_path $data_path --val_data_path $val_data_path --num_workers $num_workers"
                            elif [ "$submitit" = "True" ]; then
                                cmd="python3 submitit_finetune_gen.py --mem_per_task $mem_per_task --ngpus $world_size --nodes $nodes --output_projection $output_projection --downstream_task $downstream_task --seed $sd --patience $patience --crop_lower_bnd $crop_lower_bnd --crop_upper_bnd $crop_upper_bnd --max_delta $max_delta --jitter_sigma $jitter_sigma --rescaling_sigma $rescaling_sigma --ft_surr_phase_noise $ft_surr_phase_noise --input_channels $input_channels --input_variates $input_variates --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --ncc_weight $ncc --cos_weight $cos_weight --model $model --batch_size $bs --epochs $epochs --accum_iter $acc_it --mask_ratio $mr --weight_decay $wd --blr $blr --warmup_epochs $warmup_epochs --data_path $data_path --val_data_path $val_data_path --num_workers $num_workers"
                            else
                                cmd="torchrun --rdzv-endpoint=localhost:$port --nproc_per_node $world_size --nnodes $nodes --node_rank 0 main_finetune_gen.py --world_size $world_size --dist_eval --output_projection $output_projection --downstream_task $downstream_task --seed $sd --patience $patience --crop_lower_bnd $crop_lower_bnd --crop_upper_bnd $crop_upper_bnd --max_delta $max_delta --jitter_sigma $jitter_sigma --rescaling_sigma $rescaling_sigma --ft_surr_phase_noise $ft_surr_phase_noise --input_channels $input_channels --input_variates $input_variates --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --ncc_weight $ncc --cos_weight $cos_weight --model $model --batch_size $bs --epochs $epochs --accum_iter $acc_it --mask_ratio $mr --weight_decay $wd --blr $blr --warmup_epochs $warmup_epochs --data_path $data_path --val_data_path $val_data_path --num_workers $num_workers"
                            fi
                                    
                            if [ "$test" = "True" ]; then
                                cmd=$cmd" --test --test_data_path $test_data_path"
                            fi

                            if [ "$univariate" = "True" ]; then
                                cmd=$cmd" --univariate"
                            fi

                            if [ ! -z "$eval_criterion" ]; then
                                cmd=$cmd" --eval_criterion $eval_criterion"
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
                                cmd=$cmd" --wandb --wandb_entity $wandb_entity --wandb_project $wandb_project"
                                if [ ! -z "$wandb_id" ]; then
                                    cmd=$cmd" --wandb_id $wandb_id"
                                fi
                            fi

                            if [ "$save_output" = "True" ]; then
                                cmd=$cmd" --output_dir $output_dir"
                            fi

                            if [ ! -z "$eval_ckpt" ]; then
                                cmd=$cmd" --eval --resume $output_dir"/"$eval_ckpt"
                            fi

                            if [ ! -z "$resume" ]; then
                                cmd=$cmd" --resume $resume"
                            fi

                            echo $cmd && $cmd
                            
                            remove_cmd="rm -r "$output_dir
                            echo $remove_cmd && $remove_cmd

                        done
                    done

                done
            done
        done

    done
done