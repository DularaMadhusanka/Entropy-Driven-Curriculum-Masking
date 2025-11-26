#Entropy-Driven Curriculum Masking

This project implements a novel Curriculum Learning strategy for fine-grained image classification. Unlike standard approaches that mask "edges" (Gradient Magnitude), we propose masking "texture" (Local Entropy/Complexity).

We validate this on the Oxford-IIIT Pet Dataset, demonstrating that Entropy-based masking forces the model to learn more robust features for animals with complex fur patterns.

Author: Dulara Madusanka

Install dependencies:

pip install -r requirements.txt


Data Setup:
Ensure the Oxford-IIIT Pet dataset images are in ./data/images/.

Usage

1. Train the Novel Method (Entropy)

python main.py --dataset oxford --mask_metric entropy --num_epochs 100 --lr 0.01


2. Train the Baseline (Gradient)

python main.py --dataset oxford --mask_metric gradient --num_epochs 100 --lr 0.01


3. Compare Models

Generate the comparison charts and metrics table:

python compare_models.py \
  --baseline saved_models/r18_oxford_gradient.pth \
  --ours saved_models/r18_oxford_entropy.pth


4. Visualize Maps

See what the model "sees" (Texture vs Edges):

python visualize_maps.py


Project Structure

main.py: Entry point for training.

data_handlers.py: Contains the GPU Scorer and Dataset logic.

resnet_train.py: Training loop with curriculum injection.

models/: ResNet architecture definitio
