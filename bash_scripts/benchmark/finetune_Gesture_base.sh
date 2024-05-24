#!/usr/bin/bash
# Fine tuning 

# Basic parameters seed = [0, 101, 202, 303, 404]
seed=(0)
num_workers="8"    # number of CPUs

path="tower"        # [tower, server]
submitit="False"    # only for training on server

nodes="1"
world_size="1"      # number of GPUs
mem_per_task="96"   # memory per GPU
port="29416"

batch_size=(32)
accum_iter=(1)

epochs="75"
warmup_epochs="10"

# Callback parameters
patience="-1"
max_delta="0.0"

eval_criterion="avg"

# Model parameters
model_size="baseDeep"
model="vit_"$model_size"_patchX"

univariate="False"

# Pretraining specifications
from_scratch="False"

ignore_pos_embed_y="False"
freeze_pos_embed_y="False"

input_channels="1"

patch_height="1"
patch_width=(24)

# Pooling strategy
global_pool=(False)
attention_pool=(True)

# Augmentation parameters
masking_blockwise="False"
mask_ratio="0.00"
mask_c_ratio="0.00"
mask_t_ratio="0.00"

crop_lower_bnd="0.8"
crop_upper_bnd="1.0"

jitter_sigma="0.2"
rescaling_sigma="0.5"
ft_surr_phase_noise="0.075"

drop_path=(0.1 0.2)
layer_decay=(0.25 0.5 0.75)

# Optimizer parameters
blr=(1e-3 3e-3 1e-2) # 3e-5 if from scratch
min_lr="0.0"
weight_decay=(0.1 0.2)

# Criterion parameters
smoothing=(0.1 0.2)

# Output path
# folder="Epilepsy"
# nb_classes="2"
# input_electrodes="1"
# time_steps="168"

# folder="FD-B"
# nb_classes="3"
# input_electrodes="1"
# time_steps="5112"

folder="Gesture"
nb_classes="8"
input_electrodes="3"
time_steps="192"

# folder="EMG"
# nb_classes="3"
# input_electrodes="1"
# time_steps="1488"

# Log specifications
save_output="True"
wandb="True"
wandb_entity="oturgut"
wandb_project="OTiS_"$folder"_Classification"
wandb_id=""

plot_attention_map="False"
plot_embeddings="False"
save_embeddings="False"
save_logits="False"

# Data path
if [ "$path" = "tower" ]; then
    data_base="/home/oturgut/data/processed/benchmarks/classification/"$folder
    checkpoint_base="/home/oturgut/SiT"
else
    data_base="/vol/aimspace/users/tuo/data/processed/benchmarks/classification/"$folder
    checkpoint_base="/vol/aimspace/users/tuo/SiT"
fi

# Dataset parameters
# Training
data_path=$data_base"/train.pt"
labels_path=$data_base"/train_labels.pt"
downstream_task="classification"

# Validation
val_data_path=$data_base"/val.pt"
val_labels_path=$data_base"/val_labels.pt"

# Test
test="True"
test_data_path=$data_base"/test.pt"
test_labels_path=$data_base"/test_labels.pt"

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
        for lr in "${blr[@]}"
        do
            for ld in "${layer_decay[@]}"
            do

                for dp in "${drop_path[@]}"
                do 
                    for wd in "${weight_decay[@]}"
                    do
                        for smth in "${smoothing[@]}"
                        do

                            subfolder=("seed$sd/"$model_size"/t"$time_steps"/p"$patch_height"x"$patch_width"/ld"$ld"/dp"$dp"/smth"$smth"/wd"$wd)

                            if [ "$univariate" = "True" ]; then
                                subfolder="univariate/"$subfolder
                            else
                                subfolder="multivariate/"$subfolder
                            fi

                            if [ "$global_pool" = "True" ]; then
                                subfolder=$subfolder"/gap"
                            elif [ "$attention_pool" = "True" ]; then
                                subfolder=$subfolder"/ap"
                            else
                                subfolder=$subfolder"/cls"
                            fi

                            # OTiS
                            if [ "$model_size" = "baseDeep" ]; then
                                if [ "$path" = "tower" ]; then
                                    finetune="/home/oturgut/SiT/output/pre/otis/base/dec160d4b/p1x24/pre_b2624_blr3e-5/checkpoint-99-ncc-0.8685.pth"
                                else
                                    finetune="/vol/aimspace/users/tuo/SiT/output/pre/otis/ticorp/cos_weight0.0/ncc_weight0.1/seed0/baseDeep_dec160d4b/t1008/p1x24/wd0.15/m0.75/pre_b2624_blr3e-5/checkpoint-99-ncc-0.8685.pth"
                                    # finetune="/vol/aimspace/users/tuo/SiT/output/gen/otis/single/cos_weight0.0/ncc_weight0.1/seed0/baseDeep_dec160d4b/t1008/p1x24/wd0.15/m0.75/pre_b1_blr1e0/checkpoint-96-mse-0.1988.pth"
                                    # finetune="/vol/aimspace/users/tuo/SiT/output/pre/otis/ticorp/cos_weight0.0/ncc_weight0.1/seed0/baseDeep_dec160d4b/t1008/p1x24/wd0.15/m0.75/pre_b2624_blr1e-5/checkpoint-99-ncc-0.8662.pth"
                                fi
                            elif [ "$model_size" = "largeDeep" ]; then
                                if [ "$path" = "tower" ]; then
                                    finetune="/home/oturgut/SiT/output/pre/otis/large/dec160d4b/p1x24/pre_b768_blr3e-5/checkpoint-96-ncc-0.8667.pth"
                                else
                                    finetune="/vol/aimspace/users/tuo/SiT/output/pre/otis/ticorp/cos_weight0.0/ncc_weight0.1/seed0/largeDeep_dec160d4b/t1008/p1x24/wd0.15/m0.75/pre_b768_blr3e-5/checkpoint-96-ncc-0.8667.pth"
                                fi
                            else
                                # huge
                                if [ "$path" = "tower" ]; then
                                    finetune="/home/oturgut/SiT/output/pre/otis/huge/dec160d4b/p1x24/pre_b1680_blr1e-5/checkpoint-98-ncc-0.8661.pth"
                                else
                                    finetune="/vol/aimspace/users/tuo/SiT/output/pre/otis/ticorp/cos_weight0.0/ncc_weight0.1/seed0/hugeDeep_dec160d4b/t1008/p1x24/wd0.15/m0.75/pre_b1680_blr1e-5/checkpoint-98-ncc-0.8661.pth"
                                fi
                            fi

                            output_dir=$checkpoint_base"/output/fin/"$folder"/"$subfolder"/fin_b"$(($bs*$accum_iter*$world_size))"_blr"$lr

                            # resume=$checkpoint_base"/output/fin/"$folder"/"$subfolder"/fin_b"$bs"_blr"$lr"/checkpoint-4-pcc-0.54.pth"

                            if [ "$path" = "tower" ]; then
                                cmd="python3 main_finetune.py --seed $sd --downstream_task $downstream_task --crop_lower_bnd $crop_lower_bnd --crop_upper_bnd $crop_upper_bnd --jitter_sigma $jitter_sigma --rescaling_sigma $rescaling_sigma --ft_surr_phase_noise $ft_surr_phase_noise --input_channels $input_channels --input_electrodes $input_electrodes --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --model $model --batch_size $bs --epochs $epochs --patience $patience --max_delta $max_delta --accum_iter $accum_iter --drop_path $dp --weight_decay $wd --layer_decay $ld --min_lr $min_lr --blr $lr --warmup_epochs $warmup_epochs --smoothing $smth --data_path $data_path --labels_path $labels_path --val_data_path $val_data_path --val_labels_path $val_labels_path --nb_classes $nb_classes --num_workers $num_workers"
                            elif [ "$submitit" = "True" ]; then
                                cmd="python3 submitit_finetune.py --mem_per_task $mem_per_task --ngpus $world_size --nodes $nodes --seed $sd --downstream_task $downstream_task --crop_lower_bnd $crop_lower_bnd --crop_upper_bnd $crop_upper_bnd --jitter_sigma $jitter_sigma --rescaling_sigma $rescaling_sigma --ft_surr_phase_noise $ft_surr_phase_noise --input_channels $input_channels --input_electrodes $input_electrodes --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --model $model --batch_size $bs --epochs $epochs --patience $patience --max_delta $max_delta --accum_iter $accum_iter --drop_path $dp --weight_decay $wd --layer_decay $ld --min_lr $min_lr --blr $lr --warmup_epochs $warmup_epochs --smoothing $smth --data_path $data_path --labels_path $labels_path --val_data_path $val_data_path --val_labels_path $val_labels_path --nb_classes $nb_classes --num_workers $num_workers"
                            else
                                cmd="torchrun --rdzv-endpoint=localhost:$port --nproc_per_node $world_size --nnodes $nodes --node_rank 0 main_finetune.py --world_size $world_size --dist_eval --seed $sd --downstream_task $downstream_task --crop_lower_bnd $crop_lower_bnd --crop_upper_bnd $crop_upper_bnd --jitter_sigma $jitter_sigma --rescaling_sigma $rescaling_sigma --ft_surr_phase_noise $ft_surr_phase_noise --input_channels $input_channels --input_electrodes $input_electrodes --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --model $model --batch_size $bs --epochs $epochs --patience $patience --max_delta $max_delta --accum_iter $accum_iter --drop_path $dp --weight_decay $wd --layer_decay $ld --min_lr $min_lr --blr $lr --warmup_epochs $warmup_epochs --smoothing $smth --data_path $data_path --labels_path $labels_path --val_data_path $val_data_path --val_labels_path $val_labels_path --nb_classes $nb_classes --num_workers $num_workers"
                            fi
                            
                            if [ "$test" = "True" ]; then
                                cmd=$cmd" --test --test_data_path $test_data_path --test_labels_path $test_labels_path"
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

                            if [ "$downstream_task" = "regression" ]; then    
                                cmd=$cmd" --lower_bnd $lower_bnd --upper_bnd $upper_bnd"
                            fi

                            if [ "$masking_blockwise" = "True" ]; then
                                cmd=$cmd" --masking_blockwise --mask_c_ratio $mask_c_ratio --mask_t_ratio $mask_t_ratio"
                            fi
                            
                            if [ "$from_scratch" = "False" ]; then
                                cmd=$cmd" --finetune $finetune"
                            fi

                            if [ ! -z "$pos_label" ]; then
                                cmd=$cmd" --pos_label $pos_label"
                            fi

                            if [ ! -z "$labels_mask_path" ]; then
                                cmd=$cmd" --labels_mask_path $labels_mask_path"
                            fi

                            if [ ! -z "$val_labels_mask_path" ]; then
                                cmd=$cmd" --val_labels_mask_path $val_labels_mask_path"
                            fi

                            if [ "$global_pool" = "True" ]; then
                                cmd=$cmd" --global_pool"
                            fi

                            if [ "$attention_pool" = "True" ]; then
                                cmd=$cmd" --attention_pool"
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

                            if [ "$plot_attention_map" = "True" ]; then
                                cmd=$cmd" --plot_attention_map"
                            fi

                            if [ "$plot_embeddings" = "True" ]; then
                                cmd=$cmd" --plot_embeddings"
                            fi

                            if [ "$save_embeddings" = "True" ]; then
                                cmd=$cmd" --save_embeddings"
                            fi

                            if [ "$save_logits" = "True" ]; then
                                cmd=$cmd" --save_logits"
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