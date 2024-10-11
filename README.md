# OTiS
An open model for general time series analysis.

<p align="center">
  <img src="./figs/OTiS.png?raw=true" width=85%>
</p>

## Environment setup
Run the following commands from the root directory of this project to setup the environment and install the `timm` library. Note that this command block is only executed once during the initial environment setup.
```
conda env create --file envs/otis.yaml
conda activate otis
git clone https://github.com/oetu/pytorch-image-models.git
pip install -e pytorch-image-models
```

Activate the conda environment before running OTiS.
```
conda activate otis
```

## Model weights
Download pre-trained model weights from the [google drive](https://drive.google.com/drive/folders/1sMxJwvyZY7M2Z_gykcjLgIXa13iYDCEf?usp=sharing).

## Training
### Classification
Run the following command.
```
python3 main_finetune.py --num_workers $num_workers --seed $sd --downstream_task classification --nb_classes $nb_classes --input_channels $input_channels --input_variates $input_variates --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --model $model --batch_size $bs --epochs $epochs --blr $lr --warmup_epochs $warmup_epochs --data_path $data_path --labels_path $labels_path --val_data_path $val_data_path --val_labels_path $val_labels_path --output_dir $output_dir
```

For slurm, run the following command.
```
torchrun --rdzv-endpoint=localhost:$port --nproc_per_node $world_size --nnodes $nodes --node_rank 0 main_finetune.py --world_size $world_size --dist_eval --num_workers $num_workers --seed $sd --downstream_task classification --nb_classes $nb_classes --input_channels $input_channels --input_variates $input_variates --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --model $model --batch_size $bs --blr $lr --epochs $epochs --warmup_epochs $warmup_epochs --data_path $data_path --labels_path $labels_path --val_data_path $val_data_path --val_labels_path $val_labels_path --output_dir $output_dir
```

### Regression
Run the following command for a multi-output regression with N variables.
```
python3 main_finetune.py --num_workers $num_workers --seed $sd --downstream_task regression --nb_classes N --lower_bnd 0 --upper_bnd N --input_channels $input_channels --input_variates $input_variates --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --model $model --batch_size $bs --blr $lr --epochs $epochs --warmup_epochs $warmup_epochs --data_path $data_path --labels_path $labels_path --val_data_path $val_data_path --val_labels_path $val_labels_path --output_dir $output_dir
```

For slurm, run the following command.
```
torchrun --rdzv-endpoint=localhost:$port --nproc_per_node $world_size --nnodes $nodes --node_rank 0 main_finetune.py --world_size $world_size --dist_eval --num_workers $num_workers --seed $sd --downstream_task regression --nb_classes N --lower_bnd 0 --upper_bnd N --input_channels $input_channels --input_variates $input_variates --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --model $model --batch_size $bs --blr $lr --epochs $epochs --warmup_epochs $warmup_epochs --data_path $data_path --labels_path $labels_path --val_data_path $val_data_path --val_labels_path $val_labels_path --output_dir $output_dir
```

### Forecasting
Run the following command.
```
python3 main_finetune_gen.py --num_workers $num_workers --seed $sd --downstream_task forecasting --mask_ratio $mr --input_channels $input_channels --input_variate$input_variates --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --ncc_weight $ncc --model $model --batch_size $bs --blr $blr --epochs $epochs --warmup_epochs $warmup_epochs --data_path $data_path --val_data_path $val_data_path --output_dir $output_dir
```

For slurm, run the following command.
```
torchrun --rdzv-endpoint=localhost:$port --nproc_per_node $world_size --nnodes $nodes --node_rank 0 main_finetune_gen.py --num_workers $num_workers --seed $sd --downstream_task forecasting --mask_ratio $mr --input_channels $input_channels --input_variates $input_variates --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --ncc_weight $ncc --model $model --batch_size $bs --blr $blr --epochs $epochs --warmup_epochs $warmup_epochs --data_path $data_path --val_data_path $val_data_path --output_dir $output_dir
```

## Evaluation
Use the `--eval` flag. For classification tasks, e.g. run the following command.
```
python3 main_finetune.py --eval --resume $checkpoint --num_workers $num_workers --seed $sd --downstream_task classification --nb_classes $nb_classes --input_channels $input_channels --input_variates $input_variates --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --model $model --batch_size $batch_size --epochs $epochs --blr $blr --warmup_epoch $warmup_epochs --data_path $data_path --labels_path $labels_path --val_data_path $val_data_path --val_labels_path $val_labels_path --output_dir $output_dir
```

## Results

### Quantitative
<p align="center">
  <img src="./figs/discriminative_tasks.png?raw=true" width=85%>
</p>

<p align="center">
  <img src="./figs/generative_tasks.png?raw=true" width=85%>
</p>

### Qualitative
<p align="center">
  <img src="./figs/embeddings.png?raw=true" width=85%>
</p>

<p align="center">
  <img src="./figs/EEG_embeddings.png?raw=true" width=85%>
</p>


## Citation
Please cite the following work:
```
@article{turgut2024towards,
  title={Towards Generalisable Time Series Understanding Across Domains},
  author={Turgut, {\"O}zg{\"u}n and M{\"u}ller, Philip and Menten, Martin J and Rueckert, Daniel},
  journal={arXiv preprint arXiv:2410.07299},
  year={2024}
}
```

## Notice
This project includes third-party software components that are subject to their respective licenses. Detailed information including component names, licenses, and copyright holders is provided in the respective files. Please review the [LICENSE](https://github.com/oetu/otis/blob/main/LICENSE) file before using or distributing this software.
