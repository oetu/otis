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

## Training

### Classification
Run `main_finetune.py`.

### Regression
Run `main_finetune.py`.

### Forecasting
Run `main_finetune_gen.py`.

## Evaluation
Use the `--eval` flag. For classification tasks, run `main_finetune.py --eval`.

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
  <img src="./figs/EEG_embeddings.png?raw=true" width=85%>
</p>

<p align="center">
  <img src="./figs/embeddings.png?raw=true" width=85%>
</p>
