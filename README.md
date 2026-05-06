# Into the Unknown: Curriculum Negative Sampling for Extreme Classification

**CSCE 726 Course Project**

## Overview

This project investigates **curriculum negative sampling** strategies for training linear classifiers in extreme multi-class classification settings, where the number of classes is in the hundreds of thousands. We study this problem on a subset of [TreeOfLife-10M](https://huggingface.co/datasets/imageomics/TreeOfLife-10M), a biology dataset spanning **163,002 species**, using the [SCENT](https://arxiv.org/abs/2602.02877) (Stochastic Compositional ENTropic risk) loss and optimizer from [LibAUC](https://github.com/Optimization-AI/LibAUC).

The key challenge in extreme classification is that computing the full softmax over 163K classes per step is expensive. Negative sampling reduces this cost, but the choice of which negatives to include at each step significantly affects convergence. This project proposes a **curriculum approach** that smoothly transitions from easy global negatives to harder semantically-local negatives over training.

## Model checkpoints

Pre-trained checkpoints are available on [Google Drive](https://drive.google.com/drive/folders/1CdzFRg3nbV_qBy66jG0w2iTud8JbPuOr?usp=drive_link).

## Project Structure

```
.
├── train.py                    # Main training script
├── configs/                    # YAML config files for each experiment
│   ├── scent.yaml              # SCENT, all-class softmax
│   ├── sox.yaml                # SOX, all-class softmax
│   ├── scent_curriculum.yaml   # SCENT + curriculum sampling
│   ├── sox_curriculum.yaml     # SOX + curriculum sampling
│   ├── scent_curriculum_global.yaml  # SCENT + global-only sampling
│   ├── scent_curriculum_local.yaml   # SCENT + local-only sampling
├── libs/
│   ├── model.py                # LinearClassifier with subset-logit support
│   ├── sampler.py              # Negative sampling strategies + prototype building
│   ├── trainer.py              # train_one_epoch and evaluate
│   ├── dataloader.py           # FeaturesDataset and DataLoader builder
│   └── logger.py               # Logging setup
├── LibAUC/                     # LibAUC source (SCENT loss + optimizer)
├── data/
│   └── prototypes.pkl          # Precomputed class prototype embeddings
├── features/
│   └── treeoflife10m_subset/   # Pre-extracted features (train/val/test splits)
├── Extreme_Classification.ipynb  # Tutorial notebook (SCENT vs SOX baseline)
├── sampling.ipynb              # Sampling strategy exploration notebook
└── evaluation.ipynb            # Results and plots notebook
```

## Installation

```bash
# Clone the repo
git clone https://github.com/Morris88826/CSCE726_project.git
cd CSCE726_project

# Install LibAUC from local source
pip install ./LibAUC

# Install other dependencies
pip install torch numpy scikit-learn pyyaml
```

## Data Setup

Download the pre-extracted TreeOfLife-10M subset features:

```bash
# Download features (train/val splits)
gdown --folder 'https://drive.google.com/drive/folders/10cY2Azqz9Gnci-r4fXcGIpH_e5YaKXH_?usp=sharing' -O ./features
```

The features directory should have the structure:
```
features/treeoflife10m_subset/
    train/
        features.pt   # (762654, 512) float32 tensor
        labels.pt     # (762654,) int64 tensor
    val/
        features.pt
        labels.pt
```

## Usage

### Build Class Prototypes

Before training with any sampling strategy other than `all` or `batch`, generate the class prototype embeddings used for subgroup clustering:

```bash
python -m libs.sampler
```

This saves `data/prototypes.pkl` with mean-pooled, L2-normalized features per class.

### Training

```bash
python train.py --config configs/scent_curriculum.yaml
```

## Results

Baseline comparison on TreeOfLife-10M subset (163K classes, 20 epochs):

| Method | Sampler | Val Accuracy |
|---|---|---|
| SOX | batch | 20.1% |
| SOX | curriculum | 22.2% |
| SCENT | batch | 26.4% |
| SCENT | curriculum | 26.7% |
*(See `evaluation.ipynb` for full results and plots.)*

## References

```bibtex
@article{wei2026geometry,
  title={A Geometry-Aware Efficient Algorithm for Compositional Entropic Risk Minimization},
  author={Wei, Xiyuan and Zhou, Linli and Wang, Bokun and Lin, Chih-Jen and Yang, Tianbao},
  journal={arXiv preprint arXiv:2602.02877},
  year={2026}
}

@inproceedings{yuan2022compositional,
  title={Compositional Training for End-to-End Deep AUC Maximization.},
  author={Yuan, Zhuoning and Guo, Zhishuai and Chawla, Nitesh V and Yang, Tianbao},
  booktitle={ICLR},
  year={2022}
}
```
