import os, sys
import cv2
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from test import test

class Trainer:
    def __init__(self, model, train_loader, val_loader, args, build_optimizer):
        self.args = args
        self.model = model
        self.optimizer = build_optimizer(model)
        self.ce_loss = F.cross_entropy
        
        # Store device and move model there
        self.device = getattr(self.args, 'device', torch.device('cpu'))
        self.model = self.model.to(self.device)
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Track metrics per epoch
        self.train_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def train(self):
        best_accuracy = 0.
        
        for epoch in range(1, self.args.num_epochs + 1):
            print('Epoch', epoch)

            # Learning Rate Decay Schedule
            if epoch == self.args.decay_epoch and epoch < self.args.stop_decay_epoch:
                for param in self.optimizer.param_groups:
                    param['lr'] = param['lr'] / 10
                print(f"Learning rate updated to {param['lr']}")
                self.args.decay_epoch += self.args.decay_step

            # 1. Train
            train_loss, train_acc = self._train_epoch(epoch-1)
            
            # 2. Test (Validation)
            test_acc = test(self.model, self.args.device, self.val_loader)
            
            # 3. Test (Test Set - Optional, if different from Val)
            if hasattr(self.args, 'testlo') and self.args.testlo:
                test(self.model, self.args.device, self.args.testlo)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(test_acc)
            
            # Print epoch summary
            print(f'Epoch {epoch} Summary:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {test_acc:.2f}%')
            
            # Plot metrics every epoch
            self._plot_metrics(epoch)
            
            # Track best accuracy but only save at the very end
            if best_accuracy < test_acc:
                best_accuracy = test_acc
                print(f'New best accuracy: {best_accuracy:.2f}%')
        
        # Save best model at the end of all epochs
        os.makedirs("./saved_models", exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join("saved_models", self.args.model_name + ".pth"))
        print(f'Final model saved to ./saved_models/{self.args.model_name}.pth')
    
    def _plot_metrics(self, epoch):
        """Plot training and validation metrics after each epoch."""
        os.makedirs("./plots", exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Training Loss
        axes[0].plot(range(1, epoch + 1), self.train_losses, 'b-o', label='Train Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss per Epoch')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Plot 2: Accuracy (Train vs Val)
        axes[1].plot(range(1, epoch + 1), self.train_accuracies, 'g-o', label='Train Accuracy')
        axes[1].plot(range(1, epoch + 1), self.val_accuracies, 'r-o', label='Val Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training vs Validation Accuracy')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        plot_path = os.path.join("./plots", f"metrics_epoch_{epoch}.png")
        plt.savefig(plot_path, dpi=100)
        print(f'Plot saved to {plot_path}')
        plt.close()
    
    def _train_epoch(self, epoch):
        self.model.train()
        
        pbar = tqdm(self.train_loader)
        correct = 0
        processed = 0
        epoch_loss = 0.0
        steps = 0
        
        # The loader yields 3 items: (data, target, dummy_probability)
        for (data, target, probability) in pbar:
            
            data, target = data.to(self.device), target.to(self.device)

            if hasattr(self, 'scorer'):
                with torch.no_grad():
                    probability = self.scorer(data)
            else:
                probability = probability.to(self.device)

            y_pred = self.model(data, self.args.percent[epoch], probability)
            
            # 3. Loss Calculation
            loss = self.ce_loss(y_pred, target)

            # 4. Backward Pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 5. Metrics
            epoch_loss += loss.item()
            pred = y_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)
            steps += 1
        
        avg_loss = epoch_loss / steps if steps > 0 else 0
        train_acc = 100 * correct / processed if processed > 0 else 0
        
        print(f'Loss={avg_loss:.4f} Accuracy={train_acc:.2f}%')
        return avg_loss, train_acc