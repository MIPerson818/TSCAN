import torch
import torch.nn as nn
from utils.grad_reverse import GradientReverseLayer

class DomainDiscriminator(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        self.grl = GradientReverseLayer()  # 梯度反转层
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),  # 二分类：源域(0)/目标域(1)
            nn.Sigmoid()
        )

    def forward(self, feat):
        """输入特征，输出域分类概率（源域概率）"""
        feat_rev = self.grl(feat)  # 反转梯度（仅在训练时生效）
        domain_prob = self.classifier(feat_rev)
        return domain_prob