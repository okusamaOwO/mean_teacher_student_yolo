"""
Gradient Reversal Layer (GRL) for Domain Adaptation

During forward pass: identity function
During backward pass: multiply gradients by -lambda (reverses gradient direction)

This encourages the feature extractor to learn domain-invariant features
by fooling the domain discriminator.

Reference:
    Ganin et al., "Domain-Adversarial Training of Neural Networks" (2016)
"""

import torch
from torch.autograd import Function


class GradientReversalFunction(Function):
    """
    Gradient Reversal Function for domain adaptation.
    
    Forward: identity transform
    Backward: negate gradients and scale by lambda
    """
    
    @staticmethod
    def forward(ctx, x, lambda_):
        """
        Forward pass - identity function
        
        Args:
            ctx: Context object for saving tensors for backward
            x: Input tensor
            lambda_: Scaling factor for gradient reversal
            
        Returns:
            x unchanged
        """
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass - reverse and scale gradients
        
        Args:
            ctx: Context object with saved tensors
            grad_output: Gradient from subsequent layers
            
        Returns:
            Negated and scaled gradient, None for lambda_ (not differentiable)
        """
        lambda_ = ctx.lambda_
        grad_input = grad_output.neg() * lambda_
        return grad_input, None


class GradientReversalLayer(torch.nn.Module):
    """
    Gradient Reversal Layer module.
    
    Usage:
        grl = GradientReversalLayer(lambda_=1.0)
        reversed_features = grl(features)
    
    Args:
        lambda_: Scaling factor for gradient reversal (default: 1.0)
                 Can be scheduled to increase during training.
    """
    
    def __init__(self, lambda_=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_
    
    def set_lambda(self, lambda_):
        """Update the lambda value (for scheduling)"""
        self.lambda_ = lambda_
    
    def forward(self, x):
        """Apply gradient reversal"""
        return GradientReversalFunction.apply(x, self.lambda_)


def get_grl_lambda(epoch, max_epochs, gamma=10.0):
    """
    Calculate lambda for GRL scheduling.
    
    Lambda increases from 0 to 1 following a sigmoid schedule.
    This allows the model to first learn good features before
    the domain adaptation kicks in strongly.
    
    Args:
        epoch: Current epoch
        max_epochs: Total number of epochs
        gamma: Controls the steepness of the sigmoid
        
    Returns:
        Lambda value between 0 and 1
    """
    p = epoch / max_epochs
    return 2.0 / (1.0 + torch.exp(torch.tensor(-gamma * p)).item()) - 1.0
