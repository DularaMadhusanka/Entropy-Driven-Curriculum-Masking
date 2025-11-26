import torch
import torchvision
import torch.nn as nn
from einops import rearrange

class ResNet18(nn.Module):

    def __init__(self, num_classes=10, pretrained=False):
        super(ResNet18, self).__init__()
        weights = 'DEFAULT' if pretrained else None
        self.net = torchvision.models.resnet18(weights=weights)
        
        # Replace final fc layer
        in_features = self.net.fc.in_features
        self.net.fc = nn.Linear(in_features, num_classes)

    def forward(self, x, percent=None, probability=None):
        # Apply curriculum masking if both percent and probability provided
        if percent is not None and probability is not None:
            # Check if percent is a tensor or float
            if isinstance(percent, torch.Tensor):
                percent = percent.item()
            
            if percent > 0.0:
                x = self._apply_curriculum_mask(x, percent, probability)
                
        return self.net(x)
    
    def _apply_curriculum_mask(self, x, percent, probability):
        B, C, H, W = x.shape
    
        # Einops infers h = H // 4, w = W // 4
        x_patches = rearrange(x, 'b c (p1 h) (p2 w) -> b (p1 p2) c h w', p1=4, p2=4)
        
        # 2. Process Probabilities
        if not isinstance(probability, torch.Tensor):
            probability = torch.tensor(probability, device=x.device, dtype=torch.float32)
        
        # Ensure dims (B, 16)
        if probability.dim() == 1: 
            probability = probability.unsqueeze(0).repeat(B, 1)
        
    
        # percent is 0.0 to 1.0
        n_mask = int(16 * percent)
        
        if n_mask > 0:
            _, indices_to_mask = torch.topk(probability, k=n_mask, dim=1)
            mask_bool = torch.zeros(B, 16, dtype=torch.bool, device=x.device)
            mask_bool.scatter_(1, indices_to_mask, True)
            mask_bool = mask_bool.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            x_patches = x_patches * (~mask_bool)

        # 5. Reconstruct image
        x = rearrange(x_patches, 'b (p1 p2) c h w -> b c (p1 h) (p2 w)', p1=4, p2=4)
        
        return x
