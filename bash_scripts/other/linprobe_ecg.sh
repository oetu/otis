#!/usr/bin/bash
# Linear probing

# Basic parameters seed = [0, 101, 202, 303, 404]
seed=(0)
num_workers="12"    # number of CPUs

path="server"       # [tower, server]
submitit="False"     # only for training on server

nodes="1"
world_size="4"      # number of GPUs
mem_per_task="32"   # memory per GPU
port="29408"

batch_size=(1024)
accum_iter=(1)

epochs="100"
warmup_epochs="10"

# Callback parameters
patience="15"
max_delta="0.25"

# Model parameters
model_size="baseDeep"
model="vit_"$model_size"_patchX"

from_scratch="False"

input_channels="1"
input_electrodes="12"
time_steps="2500"

patch_height="1"
patch_width=(100)

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

# Optimizer parameters
blr=(1e-1)
min_lr="0.0"
weight_decay=(0.15)

# Criterion parameters
smoothing=(0.1)

# Data path
if [ "$path" = "tower" ]; then
    data_base="/home/oturgut/data/processed/UKBB"
    checkpoint_base="/home/oturgut/SiT"
else
    data_base="/vol/aimspace/projects/ukbb/data/cardiac/cardiac_segmentations/projects/ecg"
    checkpoint_base="/vol/aimspace/users/tuo/SiT"
fi

# Dataset parameters
# Training balanced
# data_path=$data_base"/ecgs_train_flutter_all_balanced_noBase_gn.pt"
# labels_path=$data_base"/labelsOneHot/labels_train_flutter_all_balanced.pt"
# downstream_task="classification"
# nb_classes="2"
# data_path=$data_base"/ecgs_train_diabetes_all_balanced_noBase_gn.pt"
# labels_path=$data_base"/labelsOneHot/labels_train_diabetes_all_balanced.pt"
# downstream_task="classification"
# nb_classes="2"
data_path=$data_base"/processed/ecgs_train_CAD_all_balanced_float32.pt"
labels_path=$data_base"/labelsOneHot/labels_train_CAD_all_balanced.pt"
downstream_task="classification"
nb_classes="2"
# data_path=$data_base"/ecgs_train_CAD_all_balanced_noBase_gn.pt"
# labels_path=$data_base"/labelsOneHot/labels_train_CAD_all_balanced.pt"
# downstream_task="classification"
# nb_classes="2"
# data_path=$data_base"/ecgs_train_Regression_noBase_gn.pt"
# labels_path=$data_base"/labelsOneHot/labels_train_Regression_stdNormed.pt"
# labels_mask_path=$data_base"/labels_train_Regression_mask.pt"
# downstream_task="regression"
# # LV
# lower_bnd="0"
# upper_bnd="6"
# nb_classes="6"
# # RV
# lower_bnd="6"
# upper_bnd="10"
# nb_classes="4"
# # WT
# lower_bnd="24"
# upper_bnd="41"
# nb_classes="17"
# # Ecc
# lower_bnd="41"
# upper_bnd="58"
# nb_classes="17"
# # Err
# lower_bnd="58"
# upper_bnd="75"
# nb_classes="17"

# Validation unbalanced
# val_data_path=$data_base"/ecgs_val_ecg_imaging_noBase_gn.pt"
# val_labels_path=$data_base"/labelsOneHot/labels_val_flutter_all.pt"
# pos_label="1"
# val_data_path=$data_base"/ecgs_val_ecg_imaging_noBase_gn.pt"
# val_labels_path=$data_base"/labelsOneHot/labels_val_diabetes_all.pt"
# pos_label="1"
val_data_path=$data_base"/processed/ecgs_val_ecg_imaging_float32.pt"
val_labels_path=$data_base"/labelsOneHot/labels_val_CAD_all.pt"
pos_label="1"
# val_data_path=$data_base"/ecgs_val_Regression_noBase_gn.pt"
# val_labels_path=$data_base"/labelsOneHot/labels_val_Regression_stdNormed.pt"
# val_labels_mask_path=$data_base"/labels_val_Regression_mask.pt"

global_pool=(True)
attention_pool=(False)

# Log specifications
save_output="True"
wandb="True"
wandb_project="MAE_ECG_CAD"
wandb_id=""

plot_attention_map="False"
plot_embeddings="False"
save_embeddings="False"
save_logits="False"

# Pretraining specifications
ignore_pos_embed_y="False"
freeze_pos_embed_y="False"

# EVALUATE
eval="False"
# As filename: State the checkpoint for the inference of a specific model
# or state the (final) epoch for the inference of all models up to this epoch
# resume=$checkpoint_base"/output/lin/"$folder"/id/"$subfolder"/lin_b"$(($batch_size*$accum_iter))"_blr"$blr"_"$pre_data"/checkpoint-89.pth"

for sd in "${seed[@]}"
do

    for bs in "${batch_size[@]}"
    do
            for lr in "${blr[@]}"
            do

                for wd in "${weight_decay[@]}"
                do
                    for smth in "${smoothing[@]}"
                    do

                        folder="test2"
                        subfolder=("seed$sd/"$model_size"/t2500/p"$patch_height"x"$patch_width"/smth"$smth"/wd"$wd"/m0.8/atp")

                        # SiT
                        # finetune="/home/oturgut/SiT/output/pre/test/cos_weight0.0/ncc_weight0.1/seed0/baseDeep_dec160d4b/t2500/p1x100/wd0.15/m0.8/pre_b512_blr1e-5/checkpoint-199-ncc-0.9484.pth"
                        finetune="/vol/aimspace/users/tuo/SiT/output/pre/refactored/cos_weight0.0/ncc_weight0.1/seed0/baseDeep_dec128d2b/t2500/p1x100/wd0.15/m0.8/pre_b512_blr1e-5/checkpoint-199-ncc-0.9455.pth"
                            
                        # finetune="/home/oturgut/SiT/output/pre/cos_weight0.0/ncc_weight0.1/seed0/tinyDeep2/t5000/p1x100/wd0.15/m0.8/pre_b128_blr3e-5/checkpoint-293-ncc-0.6461.pth"
                        # finetune="/vol/aimspace/users/tuo/SiT/output/pre/SiT/ukbb/cos_weight0.0/ncc_weight0.1/seed0/tinyDeep2/t2500/p1x100/wd0.15/m0.8/pre_b128_blr1e-4/checkpoint-299-ncc-0.9606.pth"
                        # finetune=$checkpoint_base"/output/pre/"$folder"/"$subfolder"/pre_"$pre_data"/checkpoint-399.pth"
                        # finetune=$checkpoint_base"/checkpoints/mm_v230_mae_checkpoint.pth"
                        # finetune=$checkpoint_base"/checkpoints/mm_v283_mae_checkpoint.pth"
                        # finetune=$checkpoint_base"/checkpoints/tiny/v1/checkpoint-399.pth"

                        output_dir=$checkpoint_base"/output/lin/"$folder"/"$subfolder"/lin_b"$(($bs*$accum_iter*$world_size))"_blr"$lr

                        # resume=$checkpoint_base"/output/lin/"$folder"/"$subfolder"/lin_b"$bs"_blr"$lr"_"$pre_data"/checkpoint-6-pcc-0.27.pth"

                        if [ "$path" = "tower" ]; then
                            cmd="python3 main_linprobe.py --seed $sd --downstream_task $downstream_task --crop_lower_bnd $crop_lower_bnd --crop_upper_bnd $crop_upper_bnd --jitter_sigma $jitter_sigma --rescaling_sigma $rescaling_sigma --ft_surr_phase_noise $ft_surr_phase_noise --input_channels $input_channels --input_electrodes $input_electrodes --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --model $model --batch_size $bs --epochs $epochs --patience $patience --max_delta $max_delta --accum_iter $accum_iter --weight_decay $wd --min_lr $min_lr --blr $lr --warmup_epoch $warmup_epochs --smoothing $smth --data_path $data_path --labels_path $labels_path --val_data_path $val_data_path --val_labels_path $val_labels_path --nb_classes $nb_classes --num_workers $num_workers"
                        elif [ "$submitit" = "True" ]; then
                            cmd="python3 submitit_linprobe.py --mem_per_task $mem_per_task --ngpus $world_size --nodes $nodes --seed $sd --downstream_task $downstream_task --crop_lower_bnd $crop_lower_bnd --crop_upper_bnd $crop_upper_bnd --jitter_sigma $jitter_sigma --rescaling_sigma $rescaling_sigma --ft_surr_phase_noise $ft_surr_phase_noise --input_channels $input_channels --input_electrodes $input_electrodes --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --model $model --batch_size $bs --epochs $epochs --patience $patience --max_delta $max_delta --accum_iter $accum_iter --weight_decay $wd --min_lr $min_lr --blr $lr --warmup_epoch $warmup_epochs --smoothing $smth --data_path $data_path --labels_path $labels_path --val_data_path $val_data_path --val_labels_path $val_labels_path --nb_classes $nb_classes --num_workers $num_workers"
                        else
                            cmd="torchrun --rdzv-endpoint=localhost:$port --nproc_per_node $world_size --nnodes $nodes --node_rank 0 main_linprobe.py --world_size $world_size --dist_eval --seed $sd --downstream_task $downstream_task --crop_lower_bnd $crop_lower_bnd --crop_upper_bnd $crop_upper_bnd --jitter_sigma $jitter_sigma --rescaling_sigma $rescaling_sigma --ft_surr_phase_noise $ft_surr_phase_noise --input_channels $input_channels --input_electrodes $input_electrodes --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --model $model --batch_size $bs --epochs $epochs --patience $patience --max_delta $max_delta --accum_iter $accum_iter --weight_decay $wd --min_lr $min_lr --blr $lr --warmup_epoch $warmup_epochs --smoothing $smth --data_path $data_path --labels_path $labels_path --val_data_path $val_data_path --val_labels_path $val_labels_path --nb_classes $nb_classes --num_workers $num_workers"
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
                            cmd=$cmd" --wandb --wandb_project $wandb_project"
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
        
    done

done