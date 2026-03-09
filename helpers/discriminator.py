"""
Domain Discriminator Head for Domain Adaptation

This discriminator is used to classify whether features come from
source domain (0) or target domain (1).

Combined with Gradient Reversal Layer (GRL), it encourages the 
feature extractor to learn domain-invariant representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainDiscriminator(nn.Module):
    """
    Domain Discriminator Head for adversarial domain adaptation.
    
    Architecture:
        - Global Average Pooling (to handle variable spatial sizes)
        - FC layers with BatchNorm and LeakyReLU
        - Binary classification output (source=0, target=1)
    
    Args:
        in_channels: Number of input feature channels (from backbone layer)
        hidden_dim: Hidden dimension size (default: 256)
        num_layers: Number of FC layers (default: 3)
    """
    
    def __init__(self, in_channels, hidden_dim=256, num_layers=3):
        super(DomainDiscriminator, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        
        # Global Average Pooling to handle different spatial dimensions
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Build FC layers
        layers = []
        
        # First layer: from input channels to hidden dim
        layers.append(nn.Linear(in_channels, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Middle layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Final layer: binary classification
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.fc = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Feature map tensor of shape (B, C, H, W)
            
        Returns:
            Domain logits of shape (B, 1)
        """
        # Global average pooling: (B, C, H, W) -> (B, C, 1, 1)
        x = self.gap(x)
        # Flatten: (B, C, 1, 1) -> (B, C)
        x = x.view(x.size(0), -1)
        # FC layers: (B, C) -> (B, 1)
        x = self.fc(x)
        return x


class MultiScaleDomainDiscriminator(nn.Module):
    """
    Multi-scale Domain Discriminator for handling multiple feature maps.
    
    This can be used when you want to apply domain adaptation
    at multiple scales/layers.
    
    Args:
        in_channels_list: List of input channels for each scale
        hidden_dim: Hidden dimension size (default: 256)
    """
    
    def __init__(self, in_channels_list, hidden_dim=256):
        super(MultiScaleDomainDiscriminator, self).__init__()
        
        self.discriminators = nn.ModuleList([
            DomainDiscriminator(in_ch, hidden_dim)
            for in_ch in in_channels_list
        ])
    
    def forward(self, feature_list):
        """
        Forward pass for multiple feature maps
        
        Args:
            feature_list: List of feature maps at different scales
            
        Returns:
            List of domain logits for each scale
        """
        outputs = []
        for feat, disc in zip(feature_list, self.discriminators):
            outputs.append(disc(feat))
        return outputs


def compute_domain_loss(source_features, target_features, discriminator, grl):
    """
    Compute domain adversarial loss.
    
    Args:
        source_features: Feature maps from source domain (B, C, H, W)
        target_features: Feature maps from target domain (B, C, H, W)
        discriminator: DomainDiscriminator module
        grl: GradientReversalLayer module
        
    Returns:
        domain_loss: Binary cross entropy loss for domain classification
    """
    batch_size_source = source_features.size(0)
    batch_size_target = target_features.size(0)
    
    # Create domain labels
    # Source domain = 0, Target domain = 1
    source_labels = torch.zeros(batch_size_source, 1, device=source_features.device)
    target_labels = torch.ones(batch_size_target, 1, device=target_features.device)
    
    # Apply GRL before discriminator
    source_features_rev = grl(source_features)
    target_features_rev = grl(target_features)
    
    # Get domain predictions
    source_preds = discriminator(source_features_rev)
    target_preds = discriminator(target_features_rev)
    
    # Concatenate predictions and labels
    preds = torch.cat([source_preds, target_preds], dim=0)
    labels = torch.cat([source_labels, target_labels], dim=0)
    
    # Binary cross entropy loss with logits
    domain_loss = F.binary_cross_entropy_with_logits(preds, labels)
    
    return domain_loss
