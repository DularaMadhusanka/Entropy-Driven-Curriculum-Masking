# Curriculum by Masking for Oxford-IIIT Pet

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)

This repository is an experimental reproduction/adaptation of **Curriculum by
Masking (CBM)** for fine-grained image classification on the Oxford-IIIT Pet
dataset. It includes two region-scoring options: Sobel gradient magnitude and a
local texture-complexity score.

> **Scoring terminology:** The option named `entropy` is currently implemented
> as local grayscale standard deviation, not Shannon entropy or a probability
> density estimate. The score map is pooled into a 4 × 4 grid, normalized per
> image, and used to mask the highest-scoring patches. The `gradient` option
> uses Sobel gradient magnitude in the same way.

### Scoring and masking formula

Let $I$ be the input image converted to grayscale:

$$I = 0.299R + 0.587G + 0.114B$$

For the `entropy` option, each pixel receives a local standard-deviation score computed over a $3 \times 3$ neighborhood:

$$s(x,y) = \sqrt{\max\left( \text{mean}_{3\times3}(I^2) - \text{mean}_{3\times3}(I)^2,\, 10^{-8} \right)}$$

This is a local texture-variation proxy, not Shannon entropy. For the `gradient` option, the score is Sobel gradient magnitude:

$$s(x,y) = \sqrt{G_x(x,y)^2 + G_y(x,y)^2 + 10^{-8}}$$

Each score map is average-pooled to a $4 \times 4$ grid. The 16 patch scores ($q_i$) are normalized per image:

$$p_i = \frac{q_i}{\sum_{j=1}^{16} q_j + 10^{-6}}$$

At a curriculum mask ratio $r$, the model masks $\lfloor 16r \rfloor$ patches with the largest $p_i$ scores by setting their pixels to zero. The ratio $r$ is generated for each epoch by the selected curriculum schedule.

## Repository layout

```text
.
├── main.py                 # CLI entry point; selects a registered experiment
├── arguments.py            # CLI options and curriculum mask-ratio schedule
├── runs.py                 # Registry of supported model/dataset experiments
├── resnet_experiments.py   # Oxford Pet setup, training orchestration, evaluation
├── resnet_train.py         # Training loop, metrics, checkpoint and epoch plots
├── data_handlers.py        # Oxford Pet dataset/loaders and patch difficulty scores
├── models/
│   └── resnet.py           # ResNet-18 wrapper and 4 × 4 curriculum masking
├── LICENSE
└── README.md
```

Training creates these output folders in the current working directory:

```text
saved_models/               # Final model state dictionaries
plots/
├── metrics_epoch_*.png     # Per-epoch training/validation plots
├── r18_oxford_<metric>/     # Combined training curves
└── confusion_matrix.png    # Final test-set confusion matrix
```

The output folders are generated at runtime and are not part of the source
layout. There are currently no comparison or saliency-visualization scripts in
the repository.

## Dataset

Download the Oxford-IIIT Pet images and place them under `data/oxford-iiit-pet`.
The loader looks for an `images` subdirectory first; it also accepts image
files placed directly in the dataset directory.

```text
data/
└── oxford-iiit-pet/
    └── images/
        ├── Abyssinian_1.jpg
        ├── American_bulldog_1.jpg
        └── ...
```

Images must follow the dataset's `<breed>_<number>.<extension>` naming pattern
(`.jpg`, `.jpeg`, or `.png`). The breed name is derived from the filename, so
separate train/validation/test folders and annotation files are not used by
this loader. The dataset is randomly split into 80% train, 10% validation, and
10% test subsets using seed 42.

## Setup

Use a Python environment with a PyTorch and torchvision build appropriate for
your hardware, then install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

For GPU training, install the PyTorch/torchvision builds recommended for your
CUDA version if they differ from the default pip builds.

## Training

The registered experiment is ResNet-18 on Oxford-IIIT Pet. The intended command
for the texture-complexity score is:

```bash
python main.py --dataset oxford --data ./data/oxford-iiit-pet --mask_metric entropy --num_epochs 100
```

Run the Sobel-gradient baseline with:

```bash
python main.py --dataset oxford --data ./data/oxford-iiit-pet --mask_metric gradient --num_epochs 100
```

Useful options include:

| Option | Default | Description |
| --- | --- | --- |
| `--data` | `./data/oxford-iiit-pet` | Dataset directory |
| `--dataset` | `oxford` | Dataset key registered in `runs.py` |
| `--model_name` | `resnet18` | Model key registered in `runs.py` |
| `--mask_metric` | `entropy` | Region score: `entropy` (local standard deviation) or `gradient` |
| `--num_epochs` | `100` | Number of training epochs |
| `--batch_size` | `32` | Training and evaluation batch size |
| `--max_mask_ratio` | `0.75` | Maximum fraction of the 16 patches to mask |
| `--schedule_type` | `linear_repeat` | Mask schedule: `linear_repeat`, `linear`, or `constant` |

The default `linear_repeat` schedule increases the ratio from zero to the
maximum in repeating 10-epoch cycles. The experiment sets the ResNet-18
optimizer to SGD with learning rate set by `--lr` (default `0.1`), momentum
`0.9`, and weight decay `5e-4`. After each epoch, validation accuracy is
measured; the weights with the best validation accuracy are saved to
`saved_models/r18_oxford_<metric>.pth` and used for final test-set evaluation.
The test split is not evaluated during training.

The CLI currently supports only the Oxford Pet experiment. `cifar10` is not a
registered dataset.

## Implementation flow

1. `main.py` parses arguments and looks up the selected experiment in `runs.py`.
2. `resnet_experiments.py` creates the Oxford Pet data loaders, ResNet-18, and
   `GPUDifficultyScorer`.
3. `data_handlers.py` computes and normalizes each image's 4 × 4 patch scores.
4. `models/resnet.py` masks the top-scoring patches according to the
   epoch-specific curriculum ratio.
5. `resnet_train.py` trains the classifier, selects the best checkpoint using
   validation accuracy, and writes plots.

Training and evaluation subsets use separate image transforms: random crop and
flip augmentation is applied only to training data.

## License

This project is distributed under the [MIT License](LICENSE).
