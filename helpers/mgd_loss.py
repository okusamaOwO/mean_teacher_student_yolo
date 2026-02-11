import torch
import torch.nn as nn


class MGDFeatureLoss(nn.Module):
    """Masked Generative Distillation Loss for a single feature level (P3/P4/P5).
    
    Gradient flows back through student features to update backbone/neck.
    
    Args:
        channels (int): Number of channels in feature maps.
        alpha_mgd (float): Weight of MGD loss. Default: 0.00002
        lambda_mgd (float): Mask ratio (proportion of pixels masked to 0). Default: 0.6
    """
    def __init__(self, channels, alpha_mgd=0.00002, lambda_mgd=0.6):
        super(MGDFeatureLoss, self).__init__()
        self.alpha_mgd = alpha_mgd
        self.lambda_mgd = lambda_mgd

        # Generation network: recovers teacher-like features from masked student features
        self.generation = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, preds_S, preds_T):
        """
        Args:
            preds_S (Tensor): Student feature map [B, C, H, W] — WITH gradient
            preds_T (Tensor): Teacher feature map [B, C, H, W] — detached, no grad
        Returns:
            loss (Tensor): Scalar MGD loss for this feature level.
        """
        assert preds_S.shape == preds_T.shape, \
            f"Shape mismatch: student {preds_S.shape} vs teacher {preds_T.shape}"

        N, C, H, W = preds_T.shape
        device = preds_S.device

        # Random spatial mask: lambda_mgd fraction set to 0, rest keep original
        # Shape (N,1,H,W) so same mask across all channels
        mat = torch.rand((N, 1, H, W), device=device)
        mat = torch.where(mat > 1 - self.lambda_mgd,
                          torch.zeros_like(mat),
                          torch.ones_like(mat))

        # Mask student features — gradient still flows through unmasked positions
        masked_fea = torch.mul(preds_S, mat)

        # Generation network recovers full feature from masked input
        new_fea = self.generation(masked_fea)

        # MSE between recovered features and teacher features
        dis_loss = nn.functional.mse_loss(new_fea, preds_T, reduction='sum') / N

        return dis_loss * self.alpha_mgd


class MGDLoss(nn.Module):
    """Multi-level MGD Loss for P3, P4, P5 feature maps.
    
    Gradient flows: MGD loss -> generation network -> masked student features -> student neck/backbone
    
    Args:
        channels_list (list[int]): Channel count for each level [P3_ch, P4_ch, P5_ch].
        alpha_mgd (float): Weight of MGD loss per level. Default: 0.00002
        lambda_mgd (float): Mask ratio. Default: 0.6
    """
    def __init__(self, channels_list, alpha_mgd=0.00002, lambda_mgd=0.6):
        super(MGDLoss, self).__init__()
        self.mgd_losses = nn.ModuleList()
        for ch in channels_list:
            self.mgd_losses.append(
                MGDFeatureLoss(channels=ch, alpha_mgd=alpha_mgd, lambda_mgd=lambda_mgd)
            )

    def forward(self, student_features, teacher_features):
        """
        Args:
            student_features (list[Tensor]): [P3, P4, P5] from student — requires_grad=True
            teacher_features (list[Tensor]): [P3, P4, P5] from teacher — detached
        Returns:
            total_mgd_loss (Tensor): Sum of MGD losses across all feature levels.
        """
        assert len(student_features) == len(teacher_features) == len(self.mgd_losses)

        total_loss = 0.0
        for mgd_fn, s_feat, t_feat in zip(self.mgd_losses, student_features, teacher_features):
            total_loss = total_loss + mgd_fn(s_feat, t_feat)

        return total_loss
