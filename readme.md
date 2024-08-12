# Learning Generic Dissimilarity Representations with Large ECG Model for CVDs Detection

This repository contains the implementation for the paper **"Learning Generic Dissimilarity Representations with Large ECG Model for CVDs Detection"**. The code includes pretraining scripts for tokenizers and a complete self-supervised learning process using the Mamba2 model. The generated representations can be evaluated on various downstream tasks for cardiovascular disease (CVD) detection.

## Setup

### 1. Create a Virtual Environment

Before proceeding, create a Python virtual environment to manage dependencies:

```bash
python -m venv venv
source venv/bin/activate # On Windows use venv\Scripts\activate
```

### 2. Install Dependencies

Install the required dependencies listed in the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Pretraining

The pretraining phase consists of three scripts:

1. **`pretrainCrossTokenMulty.py`**: This script pretrains the `AssocTokenizer` to prevent gradient explosion during the self-supervised learning phase.
   
2. **`pretrainDeviatTokenMulty.py`**: This script pretrains the `DeviaTokenizer` for the same purpose.

3. **`pretrainECGLMMulty.py`**: This script performs the complete self-supervised learning using the `Mamba2` model.

## Performance Evaluation

After the pretraining and self-supervised learning phases, the generated files can be evaluated using the scripts in the `ValExps` folder. These scripts test the performance of the learned representations on various downstream classifiers for CVD detection.

## Citation

If you find this repository useful in your research, please consider citing our paper:

@article{Haibo2024learning,
title={Learning Generic Dissimilarity Representations with Large ECG Model for CVDs Detection},
author={Haibo Zheng},
journal={Journal Name},
year={2024},
}