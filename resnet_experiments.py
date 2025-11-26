import copy
import math
import os
import time
import numpy as np
import torch
import torchvision
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import arguments
# Import the unified loader from your data_handlers.py
from data_handlers import get_oxford_loaders 
from models.resnet import ResNet18
from resnet_train import Trainer 
from fibonacci import fibonacci

# --- 1. Optimizers & Schedules ---

def build_optimizer_resnet(model):
    return optim.SGD(model.parameters(), lr=0.1, weight_decay=5e-4, momentum=0.9)

# (lin_repeat is no longer used but kept for reference)
def lin_repeat():
    """Fibonacci-based Linear Repeat Schedule"""
    v = fibonacci(length=7)
    for i in range(1, len(v)):
        v[i] = math.log(v[i]) / (math.log(v[6]) / 0.4036067977500615)
    v = v[2:]
    v[0] = 0.07
    return v

# --- 2. Visualization & Evaluation ---

def plot_training_curves(trainer, save_path="plots"):
    """Plots Loss and Accuracy curves."""
    os.makedirs(save_path, exist_ok=True)
    
    if not trainer.train_losses:
        print("No training history found.")
        return

    epochs = range(1, len(trainer.train_losses) + 1)

    plt.figure(figsize=(14, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, trainer.train_losses, 'b-o', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, trainer.train_accuracies, 'g-o', label='Train Acc')
    plt.plot(epochs, trainer.val_accuracies, 'r--s', label='Val Acc')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "training_curves.png"))
    plt.close()

def evaluate_performance(model, device, test_loader, classes):
    print("\n--- Starting Final Evaluation ---")
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if len(batch) == 3:
                images, labels, _ = batch # Ignore the dummy probability
            else:
                images, labels = batch    # Standard 2-item unpacking
            images, labels = images.to(device), labels.to(device)
            # CBM Inference: Pass dummy prob
            dummy_prob = torch.zeros(images.shape[0], 16).to(device)
            
            try:
                outputs = model(images, 0.0, dummy_prob)
            except TypeError:
                outputs = model(images)
            
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # 1. Classification Report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))

    # 2. Confusion Matrix Heatmap
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(16, 14)) 
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title('Confusion Matrix - Oxford Pets')
    plt.tight_layout()
    
    save_file = "plots/confusion_matrix.png"
    os.makedirs("plots", exist_ok=True)
    plt.savefig(save_file)
    print(f"Confusion matrix saved to {save_file}")
    plt.close()

# --- 3. Experiments ---

def train_oxford(args):
    """
    Experiment for Oxford-IIIT Pet Dataset using GPU Scoring.
    """
    # 1. Configure Args
    args.num_classes = 37
    # Override defaults if not set
    if not hasattr(args, 'num_epochs'): args.num_epochs = 100
    
    # Update Decay Schedule for 100 epochs
    args.decay_epoch = 30
    args.decay_step = 20
    args.stop_decay_epoch = 90
    
    if not hasattr(args, 'mask_metric'):
        args.mask_metric = 'gradient'
        
    args.model_name = f'r18_oxford_{args.mask_metric}'

    if hasattr(args, 'percent'):
        while len(args.percent) < args.num_epochs:
            args.percent.append(args.percent[-1])
    else:
        # Fallback if args.percent missing entirely
        print("WARNING: args.percent missing. Generating default linear schedule.")
        args.percent = [0.75] * args.num_epochs
    
    print(f"\n>>> STARTING OXFORD PETS TRAINING")
    print(f"Metric: {args.mask_metric.upper()}")
    print(f"Schedule Length: {len(args.percent)}")
    print(f"Schedule (First 5): {[round(x,2) for x in args.percent[:5]]} ...")

    # 3. Get Loaders
    train_loader, val_loader, test_loader = get_oxford_loaders(
        dataset_path=args.data,
        batch_size=args.batch_size,
        resize=224
    )
    args.testlo = test_loader

    # 4. Setup Model
    print("Initializing ResNet18...")
    model = ResNet18(num_classes=args.num_classes)
    model = model.to(args.device)

    # 5. Initialize Trainer
    trainer = Trainer(model, train_loader, val_loader, args, build_optimizer_resnet)

    # Initialize GPU Scorer
    from data_handlers import GPUDifficultyScorer
    
    print(f"Initializing GPU Scorer ({args.mask_metric}) on {args.device}...")
    scorer = GPUDifficultyScorer(metric=args.mask_metric, patch_count=4).to(args.device)
    trainer.scorer = scorer
    
    # 6. Train
    start_time = time.time()
    trainer.train()
    total_time = time.time() - start_time
    print(f"Training finished in {total_time/60:.2f} minutes.")

    # 7. Evaluate
    print(f"\nGenerating plots for {args.model_name}...")
    plot_training_curves(trainer, save_path=f"plots/{args.model_name}")
    
    if hasattr(train_loader.dataset, 'dataset'):
        class_names = train_loader.dataset.dataset.classes
    else:
        class_names = train_loader.dataset.classes

    evaluate_performance(trainer.model, trainer.device, test_loader, class_names)

if __name__ == "__main__":
    from arguments import get_args
    
    args = get_args()
    
    # Override for testing
    args.dataset = 'oxford'
    args.num_epochs = 100
    args.gpu = 0 
    
    train_oxford(args)