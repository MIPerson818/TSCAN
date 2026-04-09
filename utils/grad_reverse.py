import torch
from torch.autograd import Function

class GradReverse(Function):
    """梯度反转层：反向传播时梯度取反，用于域对抗训练"""
    @staticmethod
    def forward(ctx, x, lambd=1.0):
        ctx.lambd = lambd
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * (-ctx.lambd), None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)