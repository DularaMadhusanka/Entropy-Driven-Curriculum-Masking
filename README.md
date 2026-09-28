# Entropy-Driven Curriculum Masking

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)

An empirical research project introducing a novel Curriculum Learning strategy for fine-grained image classification. 

Unlike conventional curriculum masking approaches that focus on edge detection via **Gradient Magnitude**, this method masks image regions based on **Local Entropy (Texture Complexity)**. By progressively obscuring high-entropy regions, the model is forced to learn robust, generalized representations—particularly effective for fine-grained datasets with complex visual patterns.

---

## 💡 Overview

| Feature | Baseline (Gradient Masking) | Proposed Method (Entropy Masking) |
| :--- | :--- | :--- |
| **Focus Area** | Sharp edges and object boundaries | Spatial texture complexity and fine-grained patterns |
| **Masking Metric** | Sobel Gradient Magnitude | Local Spatial Entropy |
| **Primary Use Case** | General object localization | Fine-grained classification (e.g., animal fur patterns) |

Validated on the **Oxford-IIIT Pet Dataset**, this strategy prevents the network from over-relying on easy texture cues, driving better feature acquisition across complex, visually similar classes.

---

## ✨ Features

- **Texture-Driven Saliency Scoring:** GPU-accelerated local entropy calculation targeting regions of rich visual texture.
- **Curriculum Masking Engine:** Dynamic masking schedule injected directly into the model training loop.
- **Comparative Benchmarking:** Built-in tools to evaluate performance against standard gradient-based baselines.
- **Saliency Map Visualization:** Diagnostic tools to visually inspect and contrast texture-based vs. edge-based masking regions.

---

## 📂 Repository Structure

```text
.
├── main.py              # Main training entry point
├── compare_models.py    # Evaluation script for baseline vs. proposed comparison
├── visualize_maps.py    # Visual inspection tool for mask generation
├── data_handlers.py     # Custom Dataset logic & GPU-accelerated entropy/gradient scorers
├── resnet_train.py     # Training engine with curriculum schedule injection
├── requirements.txt     # Environment dependencies
├── data/
│   └── images/          # Target dataset directory (Oxford-IIIT Pet images)
└── models/
    └── resnet.py        # ResNet architecture with masking integration
```

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/DularaMadhusanka/Entropy-Driven-Curriculum-Masking.git
   cd Entropy-Driven-Curriculum-Masking
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Data Setup:**
   Ensure the Oxford-IIIT Pet dataset images are downloaded and placed in the `./data/images/` directory.

---

## 🚀 Usage

### 1. Train Proposed Method (Entropy Masking)
Train the ResNet architecture using local entropy to guide the masking curriculum:

```bash
python main.py --dataset oxford --mask_metric entropy --num_epochs 100 --lr 0.01
```

### 2. Train Baseline (Gradient Masking)
Train the benchmark model using standard gradient magnitude masking:

```bash
python main.py --dataset oxford --mask_metric gradient --num_epochs 100 --lr 0.01
```

### 3. Compare Model Performance
Generate comparative evaluation charts and summary metric tables between the two trained models:

```bash
python compare_models.py \
  --baseline saved_models/r18_oxford_gradient.pth \
  --ours saved_models/r18_oxford_entropy.pth
```

### 4. Visualize Saliency Maps
Inspect and compare what the model targets during training (Texture Complexity vs. Edge Boundaries):

```bash
python visualize_maps.py
```

---

## 👤 Author

**Dulara Madusanka**  
- GitHub: [@DularaMadhusanka](https://github.com/DularaMadhusanka)
