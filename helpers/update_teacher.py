import torch

def de_parallel(model):
    """De-parallelize a model: returns single-GPU model if model is of type DP or DDP."""
    return model.module if hasattr(model, 'module') else model

@torch.no_grad()
def update_teacher(student_model, teacher_model, alpha):
    """
    Update teacher model using exponential moving average.
    Updates both parameters (weights) and buffers (BatchNorm stats).
    Handles DDP/DataParallel wrapped models automatically.
    
    Args:
        student_model: Student model (may be DDP wrapped)
        teacher_model: Teacher model (should NOT be DDP wrapped)
        alpha: EMA decay rate (higher = slower update, e.g., 0.999)
    """
    # Handle DDP/DataParallel wrapped models
    student = de_parallel(student_model)
    teacher = de_parallel(teacher_model)
    
    # Update parameters (weights and biases)
    for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
        teacher_param.data.mul_(alpha).add_(student_param.data, alpha=1 - alpha)
    
    # Update buffers (BatchNorm running_mean and running_var)
    for teacher_buf, student_buf in zip(teacher.buffers(), student.buffers()):
        if teacher_buf.dtype == torch.long or teacher_buf.dtype == torch.int:
            teacher_buf.data.copy_(student_buf.data)
        else:
            teacher_buf.data.mul_(alpha).add_(student_buf.data, alpha=1 - alpha)
