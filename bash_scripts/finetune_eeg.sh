#!/usr/bin/bash
# Fine tuning 

# Basic parameters seed = [0, 101, 202, 303, 404]
seed=(0)
batch_size=(32)
accum_iter=(1)

epochs="200"
warmup_epochs="5"

# Callback parameters
patience="15"
max_delta="0.15" # for AUROC

# Model parameters
input_channels="1"
input_electrodes="30"
time_steps="6000"

model_size="tinyDeep"
model="vit_"$model_size"_patchX"

patch_height="1"
patch_width=(100)

# Augmentation parameters
masking_blockwise="False"
mask_ratio="0.00"
mask_c_ratio="0.00"
mask_t_ratio="0.00"

jitter_sigma="0.2"
rescaling_sigma="0.5"
ft_surr_phase_noise="0.075"

drop_path=(0.1)
layer_decay=(0.5)

# Optimizer parameters
blr=(3e-5) # 3e-5 if from scratch
min_lr="0.0"
weight_decay=(0.1)

# Criterion parameters
smoothing=(0.1)

folds=(0) # 1 2 3 4 5 6 7 8 9)
for fold in "${folds[@]}"
do

    # Data path
    path="server"
    if [ "$path" = "tower" ]; then
        data_base="/home/oturgut/data/processed/lemon/kfold/fold"$fold
        checkpoint_base="/home/oturgut/SiT"
    else
        data_base="/vol/aimspace/users/tuo/data/lemon/kfold/fold"$fold
        checkpoint_base="/vol/aimspace/users/tuo/SiT"
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
    val_data_path=$data_base"/data_val.pt"
    val_labels_path=$data_base"/labels_val_stdNormed.pt"
    # val_labels_mask_path=$data_base"/labels_val_Regression_mask.pt"

    global_pool=(True)
    attention_pool=(False)
    num_workers="24"

    # Log specifications
    save_output="False"
    wandb="True"
    wandb_project="MAE_EEG_Age"
    wandb_id=""

    plot_attention_map="False"
    plot_embeddings="False"
    save_embeddings="False"
    save_logits="False"

    # Pretraining specifications
    pre_batch_size=(128)
    pre_blr=(1e-5)
    ignore_pos_embed_y="False"
    trainable_pos_embed_y="True"

    from_scratch="False"

    # EVALUATE
    eval="False"
    # As filename: State the checkpoint for the inference of a specific model
    # or state the (final) epoch for the inference of all models up to this epoch
    #resume=$checkpoint_base"/output/fin/"$folder"/id/"$subfolder"/fin_b"$(($batch_size*$accum_iter))"_blr"$blr"_"$pre_data"/checkpoint-89.pth"

    for sd in "${seed[@]}"
    do

        for bs in "${batch_size[@]}"
        do
            for ld in "${layer_decay[@]}"
            do
                for lr in "${blr[@]}"
                do

                    for dp in "${drop_path[@]}"
                    do 
                        for wd in "${weight_decay[@]}"
                        do
                            for smth in "${smoothing[@]}"
                            do

                                folder="lemon/kfold/fold"$fold"/eeg/Age/SiT"
                                subfolder="seed$sd/"$model_size"/t"$time_steps"/p"$patch_height"x"$patch_width"/ld"$ld"/dp"$dp"/smth"$smth"/wd"$weight_decay"/m0.8"

                                pre_data="b"$pre_batch_size"_blr"$pre_blr
                                # finetune=$checkpoint_base"/output/pre/"$folder"/"$subfolder"/pre_"$pre_data"/checkpoint-399.pth"
                                
                                # finetune=$checkpoint_base"/output/pre/tuh/eeg/all/15ch/ncc_weight0.1/seed0/tiny/t3000/p1x100/wd0.15/m0.8/pre_b256_blr1e-5/checkpoint-196-ncc-0.78.pth"

                                # finetune="/vol/aimspace/users/tuo/SiT/output/pre/TempEncoder/cos_weight0.0/ncc_weight0.1/seed0/tinyDeep2/t2500/p1x100/wd0.15/m0.8/pre_b128_blr3e-5/checkpoint-173-ncc-0.8602.pth"
                                # finetune="/vol/aimspace/users/tuo/SiT/output/pre/TempEncoder/noWLoss/cos_weight0.0/ncc_weight0.1/seed0/tinyDeep2/t2500/p1x100/wd0.15/m0.8/pre_b128_blr3e-5/checkpoint-198-ncc-0.8919.pth"

                                # finetune="/vol/aimspace/users/tuo/SiT/output/pre/fresh/WLoss/posEncX60/cos_weight0.0/ncc_weight0.1/seed0/tinyDeep2/t2500/p1x100/wd0.15/m0.8/pre_b768_blr1e-5/checkpoint-198-ncc-0.8828.pth"

                                # finetune="/vol/aimspace/users/tuo/SiT/output/pre/fresh/noTempEncoder/WLoss/cos_weight0.0/ncc_weight0.1/seed0/tinyDeep2/t2500/p1x100/wd0.15/m0.8/pre_b768_blr1e-5/checkpoint-198-ncc-0.8803.pth"
                                # finetune="/vol/aimspace/users/tuo/SiT/output/pre/fresh/noTempEncoder/noWLoss/cos_weight0.0/ncc_weight0.1/seed0/tinyDeep2/t2500/p1x100/wd0.15/m0.8/pre_b768_blr1e-5/checkpoint-198-ncc-0.9177.pth"

                                finetune="/vol/aimspace/users/tuo/SiT/output/pre/fresh/WLoss/cos_weight0.0/ncc_weight0.1/seed0/tinyDeep2/t2500/p1x100/wd0.15/m0.8/pre_b768_blr1e-5/checkpoint-198-ncc-0.8827.pth"
                                # finetune="/vol/aimspace/users/tuo/SiT/output/pre/fresh/noWLoss/cos_weight0.0/ncc_weight0.1/seed0/tinyDeep2/t2500/p1x100/wd0.15/m0.8/pre_b768_blr1e-5/checkpoint-198-ncc-0.9158.pth"

                                # finetune="/home/oturgut/SiT/output/pre/noWLoss/cos_weight0.0/ncc_weight0.1/seed0/tinyDeep2/t2500/p1x100/wd0.15/m0.8/pre_b128_blr3e-5/checkpoint-98-ncc-0.8766.pth"
                                # finetune="/home/oturgut/SiT/output/pre/TempEncoder/cos_weight0.0/ncc_weight0.1/seed0/tinyDeep/t2500/p1x100/wd0.15/m0.8/pre_b576_blr1e-5/checkpoint-198-ncc-0.9233.pth"
                                # finetune="/home/oturgut/SiT/output/pre/wLoss/cos_weight0.0/ncc_weight0.1/seed0/tinyDeep2/t5000/p1x100/wd0.15/m0.8/pre_b128_blr3e-5/checkpoint-242-ncc-0.8882.pth"
                                # finetune="/home/oturgut/SiT/output/pre/cos_weight0.0/ncc_weight0.1/seed0/tinyDeep/t2500/p1x100/wd0.15/m0.8/pre_b576_blr1e-5/checkpoint-199-ncc-0.8946.pth"
                                # finetune="/home/oturgut/SiT/output/pre/test4/cos_weight0.0/ncc_weight0.1/seed0/tinyDeep2/t5000/p1x100/wd0.15/m0.8/pre_b128_blr3e-5/checkpoint-280-ncc-0.6527.pth"

                                # finetune="/home/oturgut/mae/output/pre/siggy/cos_weight0.1/ncc_weight0.1/seed0/tinyDeep2/t2500/p1x100/wd0.15/m0.8/pre_b1024_blr1e-5/checkpoint-399-ncc-0.8839.pth"

                                # finetune="/home/oturgut/mae/output/pre/tuh/250Hz/eeg/10ch/ncc_weight0.1/seed0/tinyUp/t3000/p1x100/wd0.15/m0.8/pre_b256_blr1e-5/checkpoint-199-ncc-0.8074.pth"
                                # finetune="/home/oturgut/mae/output/pre/tuh/250Hz/eeg/ncc_weight0.1/seed0/tiny/t3000/p1x100/wd0.15/m0.8/pre_b256_blr1e-5/checkpoint-199-ncc-0.8500.pth"
                                # finetune="/home/oturgut/mae/checkpoints/mm_v230_mae_checkpoint.pth"
                                # finetune="/home/oturgut/mae/checkpoints/tiny/v1/checkpoint-399.pth"
                                # finetune="/home/oturgut/mae/output/pre/lemon/full/eeg/ncc_weight0.1/seed0/tiny/t6000/p1x100/wd0.15/m0.8/pre_b64_blr1e-4/checkpoint-196-ncc-0.81.pth"

                                output_dir=$checkpoint_base"/output/fin/"$folder"/"$subfolder"/fin_b"$(($bs*$accum_iter))"_blr"$lr"_"$pre_data

                                # resume=$checkpoint_base"/output/fin/"$folder"/"$subfolder"/fin_b"$bs"_blr"$lr"_"$pre_data"/checkpoint-4-pcc-0.54.pth"

                                cmd="python3 main_finetune.py --seed $sd --downstream_task $downstream_task --jitter_sigma $jitter_sigma --rescaling_sigma $rescaling_sigma --ft_surr_phase_noise $ft_surr_phase_noise --input_channels $input_channels --input_electrodes $input_electrodes --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --model $model --batch_size $bs --epochs $epochs --patience $patience --max_delta $max_delta --accum_iter $accum_iter --drop_path $dp --weight_decay $wd --layer_decay $ld --min_lr $min_lr --blr $lr --warmup_epoch $warmup_epochs --smoothing $smth --data_path $data_path --labels_path $labels_path --val_data_path $val_data_path --val_labels_path $val_labels_path --nb_classes $nb_classes --num_workers $num_workers"
                                
                                if [ "$ignore_pos_embed_y" = "True" ]; then
                                    cmd=$cmd" --ignore_pos_embed_y"
                                fi

                                if [ "$trainable_pos_embed_y" = "True" ]; then
                                    cmd=$cmd" --trainable_pos_embed_y"
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
        done

    done

done