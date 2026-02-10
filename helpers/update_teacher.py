import torch

def update_teacher(student_model, teacher_model, alpha):
    """
    Update teacher model using exponential moving average.
    Updates both parameters (weights) and buffers (BatchNorm stats).
    """
    # Update parameters (weights and biases)
    for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
        teacher_param.data.mul_(alpha).add_(student_param.data, alpha=1 - alpha)
    
    # Update buffers (BatchNorm running_mean and running_var)
    for teacher_buf, student_buf in zip(teacher_model.buffers(), student_model.buffers()):
        if teacher_buf.dtype == torch.long or teacher_buf.dtype == torch.int:
            teacher_buf.data.copy_(student_buf.data)
        else:
            teacher_buf.data.mul_(alpha).add_(student_buf.data, alpha=1 - alpha)
