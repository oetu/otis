#!/usr/bin/bash
# Inference on the linprobed model 

# Basic parameters seed = [0, 101, 202, 303, 404]
seed="0"
batch_size=(8)
accum_iter=(1)

epochs="400"
warmup_epochs="5"

# Callback parameters
patience="15"
max_delta="0.0" # for AUROC

# Model parameters
input_channels="1"
input_electrodes="12"
time_steps="2500"
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

layer_decay="0.75"

# Optimizer parameters
blr=(3e-5)
min_lr="0.0"
weight_decay=(0.2)

# Criterion parameters
smoothing=(0.0)

# Data path
path="tower"
if [ "$path" = "tower" ]; then
    data_base="/home/oturgut/data/processed/ukbb"
    checkpoint_base="/home/oturgut/mae"
else
    data_base="/vol/aimspace/projects/ukbb/data/cardiac/cardiac_segmentations/projects/ecg"
    checkpoint_base="/vol/aimspace/users/tuo/mae"
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
# data_path=$data_base"/ecgs_train_CAD_all_balanced_noBase_gn.pt"
# labels_path=$data_base"/labelsOneHot/labels_train_CAD_all_balanced.pt"
# downstream_task="classification"
# nb_classes="2"
data_path=$data_base"/ecgs_train_Regression_noBase_gn.pt"
labels_path=$data_base"/labelsOneHot/labels_train_Regression_stdNormed.pt"
labels_mask_path=$data_base"/labels_train_Regression_mask.pt"
downstream_task="regression"
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
# Ecc
lower_bnd="41"
upper_bnd="58"
nb_classes="17"
# # Err
# lower_bnd="58"
# upper_bnd="75"
# nb_classes="17"

# Validation unbalanced
# val_data_path=$data_base"/ecgs_val_ecg_imaging_noBase_gn.pt"
# val_labels_path=$data_base"/labelsOneHot/labels_val_flutter_all.pt"
# pos_label="1"
# val_data_path=$data_base"/ecgs_val_ecg_imaging_noBase_gn.pt"
# val_labels_path=$data_base"/labelsOneHot/labels_val_flutter_all.pt"
# pos_label="1"
# val_data_path=$data_base"/ecgs_val_ecg_imaging_noBase_gn.pt"
# val_labels_path=$data_base"/labelsOneHot/labels_val_diabetes_all.pt"
# pos_label="1"
# val_data_path=$data_base"/ecgs_val_ecg_imaging_noBase_gn.pt"
# val_labels_path=$data_base"/labelsOneHot/labels_val_diabetes_all.pt"
# pos_label="1"
# val_data_path=$data_base"/ecgs_val_ecg_imaging_noBase_gn.pt"
# val_labels_path=$data_base"/labelsOneHot/labels_val_CAD_all.pt"
# pos_label="1"
# val_data_path=$data_base"/ecgs_val_ecg_imaging_noBase_gn.pt"
# val_labels_path=$data_base"/labelsOneHot/labels_val_CAD_all.pt"
# pos_label="1"
# val_data_path=$data_base"/ecgs_val_Regression_noBase_gn.pt"
# val_labels_path=$data_base"/labelsOneHot/labels_val_Regression_stdNormed.pt"
# val_labels_mask_path=$data_base"/labels_val_Regression_mask.pt"
val_data_path=$data_base"/ecgs_test_Regression_noBase_gn.pt"
val_labels_path=$data_base"/labelsOneHot/labels_test_Regression_stdNormed.pt"
val_labels_mask_path=$data_base"/labels_test_Regression_mask.pt"

global_pool=(True)
attention_pool=(False)
num_workers="24"

# Log specifications
wandb="True"
wandb_project="MAE_ECG_Fin_Tiny_Ecc"
save_logits="False"

# Pretraining specifications
pre_batch_size=(128)
pre_blr=(1e-5)

folder="ukbb/ecg/Ecc/MMonly"
subfolder=("seed$seed/"$model_size"/t"$time_steps"/p"$patch_height"x"$patch_width"/ld"$layer_decay"/dp"$drop_path"/smth"$smoothing"/wd"$weight_decay"/m0.8")

pre_data="b"$pre_batch_size"_blr"$pre_blr
output_dir=$checkpoint_base"/output/fin/"$folder"/"$subfolder"/fin_b"$(($batch_size*$accum_iter))"_blr"$blr"_"$pre_data

# As filename: State the checkpoint for the inference of a specific model
# or state the (final) epoch for the inference of all models up to this epoch
resume=$checkpoint_base"/output/lin/"$folder"/"$subfolder"/lin_b"$(($batch_size*$accum_iter))"_blr"$blr"_"$pre_data"/checkpoint-55-pcc-0.32.pth"

if [ "$downstream_task" = "regression" ]; then
    cmd="python3 main_linprobe.py --device cpu --eval --resume $resume --lower_bnd $lower_bnd --upper_bnd $upper_bnd --seed $seed --downstream_task $downstream_task --mask_ratio $mask_ratio --jitter_sigma $jitter_sigma --rescaling_sigma $rescaling_sigma --ft_surr_phase_noise $ft_surr_phase_noise --input_channels $input_channels --input_electrodes $input_electrodes --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --model $model --batch_size $batch_size --epochs $epochs --patience $patience --max_delta $max_delta --accum_iter $accum_iter --weight_decay $weight_decay --layer_decay $layer_decay --min_lr $min_lr --blr $blr --warmup_epoch $warmup_epochs --smoothing $smoothing --data_path $data_path --labels_path $labels_path --val_data_path $val_data_path --val_labels_path $val_labels_path --nb_classes $nb_classes --num_workers $num_workers"
else
    cmd="python3 main_linprobe.py --device cpu --eval --resume $resume --seed $seed --downstream_task $downstream_task --mask_ratio $mask_ratio --jitter_sigma $jitter_sigma --rescaling_sigma $rescaling_sigma --ft_surr_phase_noise $ft_surr_phase_noise --input_channels $input_channels --input_electrodes $input_electrodes --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --model $model --batch_size $batch_size --epochs $epochs --patience $patience --max_delta $max_delta --accum_iter $accum_iter --weight_decay $weight_decay --layer_decay $layer_decay --min_lr $min_lr --blr $blr --warmup_epoch $warmup_epochs --smoothing $smoothing --data_path $data_path --labels_path $labels_path --val_data_path $val_data_path --val_labels_path $val_labels_path --nb_classes $nb_classes --num_workers $num_workers"
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