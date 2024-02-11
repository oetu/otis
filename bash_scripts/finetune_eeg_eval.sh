#!/usr/bin/bash
# Fine tuning

# Basic parameters seed = [0, 101, 202, 303, 404]
seed="0"
batch_size=(32)
accum_iter=(1)

epochs="200"
warmup_epochs="5"

# Callback parameters
patience="25"
max_delta="0.25" # for AUROC

# Model parameters
input_channels="1"
input_electrodes="30"
time_steps="6000"
model_size="tiny"
model="vit_"$model_size"_patchX"

patch_height="1"
patch_width="100"

# Augmentation parameters
masking_blockwise="False"
mask_ratio="0.00"
mask_c_ratio="0.00"
mask_t_ratio="0.00"

jitter_sigma="0.2"
rescaling_sigma="0.5"
ft_surr_phase_noise="0.075"

drop_path=(0.1)
layer_decay="0.75"

# Optimizer parameters
blr=(3e-5)
min_lr="0.0"
weight_decay=(0.1)

# Criterion parameters
smoothing=(0.1)

# Data path
path="tower"
if [ "$path" = "tower" ]; then
    data_base="/home/oturgut/data/processed/lemon/full"
    checkpoint_base="/home/oturgut/mae"
else
    data_base="/vol/aimspace/projects/ukbb/data/cardiac/cardiac_segmentations/projects/ecg"
    checkpoint_base="/vol/aimspace/users/tuo/mae"
fi

# Dataset parameters
# Training
data_path=$data_base"/data_train.pt"
labels_path=$data_base"/labels_train_stdNormed.pt"
# labels_mask_path=$data_base"/labels_train_Regression_mask.pt"
downstream_task="regression"
# Age
lower_bnd="0"
upper_bnd="1"
nb_classes="1"

# Validation 
val_data_path=$data_base"/data_test.pt"
val_labels_path=$data_base"/labels_test_stdNormed.pt"
# val_labels_mask_path=$data_base"/labels_val_Regression_mask.pt"

global_pool=(False)
attention_pool=(True)
num_workers="24"

# Log specifications
wandb="True"
wandb_project="MAE_EEG_Fin_Tiny_Age"
save_logits="True"

# Pretraining specifications
pre_batch_size=(128)
pre_blr=(1e-5)

folder="lemon/full/eeg/Age/ukbb_pretrained/v230"
subfolder=("seed$seed/"$model_size"/t"$time_steps"/p"$patch_height"x"$patch_width"/ld"$layer_decay"/dp"$drop_path"/smth"$smoothing"/wd"$weight_decay"/m0.8")

pre_data="b"$pre_batch_size"_blr"$pre_blr
output_dir=$checkpoint_base"/output/fin/"$folder"/"$subfolder"/fin_b"$(($batch_size*$accum_iter))"_blr"$blr"_"$pre_data

# As filename: State the checkpoint for the inference of a specific model
# or state the (final) epoch for the inference of all models up to this epoch
resume=$checkpoint_base"/output/fin/"$folder"/"$subfolder"/fin_b"$(($batch_size*$accum_iter))"_blr"$blr"_"$pre_data"/checkpoint-116-pcc-0.9040.pth"
# /home/oturgut/mae/output/fin/lemon/eeg/Age/ukbb_pretrained/new/seed0/tiny/t6000/p1x100/ld0.75/dp0.1/smth0.1/wd0.1/m0.8/fin_b32_blr3e-5_b128_blr1e-5/checkpoint-84-pcc-0.8766.pth

cmd="python3 main_finetune.py --eval --output_dir $output_dir --resume $resume --seed $seed --downstream_task $downstream_task --mask_ratio $mask_ratio --jitter_sigma $jitter_sigma --rescaling_sigma $rescaling_sigma --ft_surr_phase_noise $ft_surr_phase_noise --input_channels $input_channels --input_electrodes $input_electrodes --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --model $model --batch_size $batch_size --epochs $epochs --patience $patience --max_delta $max_delta --accum_iter $accum_iter --drop_path $drop_path --weight_decay $weight_decay --layer_decay $layer_decay --min_lr $min_lr --blr $blr --warmup_epoch $warmup_epochs --smoothing $smoothing --data_path $data_path --labels_path $labels_path --val_data_path $val_data_path --val_labels_path $val_labels_path --nb_classes $nb_classes --num_workers $num_workers"

if [ "$downstream_task" = "regression" ]; then
    cmd=$cmd" --lower_bnd $lower_bnd --upper_bnd $upper_bnd"
fi

if [ "$masking_blockwise" = "True" ]; then
    cmd=$cmd" --masking_blockwise --mask_c_ratio $mask_c_ratio --mask_t_ratio $mask_t_ratio"
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
fi

if [ "$save_logits" = "True" ]; then
    cmd=$cmd" --save_logits"
fi

echo $cmd && $cmd