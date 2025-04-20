import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    
    def __init__(self, embedding_size=128, class_num=100, s=64, m=0.50, gamma=2):
        """ArcFace formula: 
            cos(m + theta) = cos(m)cos(theta) - sin(m)sin(theta)
        Note that:
            0 <= m + theta <= Pi
        So if (m + theta) >= Pi, then theta >= Pi - m. In [0, Pi]
        we have:
            cos(theta) < cos(Pi - m)
        So we can use cos(Pi - m) as threshold to check whether 
        (m + theta) go out of [0, Pi]

        Args:
            embedding_size: usually 128, 256, 512 ...
            class_num: num of people when training
            s: scale, see normface https://arxiv.org/abs/1704.06369
            m: margin, see SphereFace, CosFace, and ArcFace paper
        """
        super(ArcFaceLoss, self).__init__()
        self.in_features = embedding_size
        self.out_features = class_num
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(class_num, embedding_size)).cuda()
        nn.init.xavier_uniform_(self.weight)
        # print('-----------',self.weight.size())

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.gamma = gamma
        self.ce = torch.nn.CrossEntropyLoss()

        def forward(self, input, target):
            logp = self.ce(input, target)
            p = torch.exp(-logp)
            loss = (1 - p) ** self.gamma * logp
            return loss.mean()

    def forward(self, input, label):
        # print('-----------',F.normalize(input).size())
        # print('-----------',nn.functional.normalize(self.weight,p=2, dim=1, eps=1e-12).size())
        # cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        cosine = nn.functional.linear(nn.functional.normalize(input,p=2, dim=1, eps=1e-12), nn.functional.normalize(self.weight,p=2, dim=1, eps=1e-12))
        
        sine = ((1.0 - cosine.pow(2)).clamp(0, 1)).sqrt()
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)  # drop to CosFace
        # update y_i by phi in cosine
        output = cosine * 1.0  # make backward works
        batch_size = len(output)
        output[range(batch_size), label] = phi[range(batch_size), label]
        loss_input = output * self.s 
        logp = self.ce(loss_input, label)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

        # return loss.mean()