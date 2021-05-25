# DECODE FISH
DECODE FISH is a deep learning based algorithm to localize fluorescent spots in RNA-FISH (smFISH) data. 
It allows for fast and precise localization even under difficult conditions of low SNR and high density.

The method is an extension of the DECODE algorithm to 3D data (https://www.biorxiv.org/content/10.1101/2020.10.26.355164v1).

## Installation
We recommend to use a [conda](https://docs.conda.io/en/latest/miniconda.html) virtual environment.
DECODE FISH requires a GPU with CUDA capability of 3.7 or higher and at least 8 GB RAM.
```bash
git clone https://github.com/TuragaLab/decode_fish.git
cd decode_fish
conda env create -f requirements.yaml
conda activate decode_fish_dev
```

## Getting started

Analyzing a data set with DECODE FISH has two steps, training the network and running the prediction.
These can be executed these scripts:

```
python decode_fish/train.py +experiment=your_experiment
python decode_fish/predict.py out_file='results.csv' model_path='your_experiment/model.pkl' image_path='your_recordings/*tif'
```

Training is parametrized by a .yaml config file. 
Please refer to the experiment.ipynb notebook to see the complete workflow of setting training parameters, 
running the training, performing predictions and inspecting the results on an example dataset. 
We recommed making a copy of this notebook for your own experiments.


