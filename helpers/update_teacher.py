import torch
@torch.no_grad()  # no gradients needed for teacher update
def update_teacher(student_model, teacher_model, alpha=0.99):
    """
    Update teacher parameters as EMA of student parameters.
    Args:
        student_model: nn.Module – your student network
        teacher_model: nn.Module – your teacher network
        alpha: float – EMA decay (0.99–0.999 typical)
    """
    for teacher_params, student_params in zip(teacher_model.parameters(), student_model.parameters()):
        teacher_params.data.mul_(alpha).add_(student_params.data, alpha=1 - alpha)
