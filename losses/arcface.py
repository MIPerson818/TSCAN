import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    def __init__(self, embedding_size=128, class_num=100, s=64.0, m=0.50, gamma=0.0):
        super().__init__()
        self.in_features = embedding_size
        self.out_features = class_num
        self.s = float(s)
        self.m = float(m)
        self.gamma = float(gamma)

        self.weight = nn.Parameter(torch.empty(class_num, embedding_size))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, input, label):
        input = F.normalize(input.float(), p=2, dim=1, eps=1e-12)
        weight = F.normalize(self.weight.float(), p=2, dim=1, eps=1e-12)
        cosine = F.linear(input, weight)

        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0.0, 1.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        output = cosine.clone()
        batch_indices = torch.arange(output.size(0), device=label.device)
        output[batch_indices, label] = phi[batch_indices, label].to(output.dtype)
        logits = output * self.s

        ce_loss = self.ce(logits, label)
        if self.gamma > 0:
            p = torch.exp(-ce_loss)
            loss = ((1.0 - p) ** self.gamma) * ce_loss
        else:
            loss = ce_loss
        return loss.mean()
