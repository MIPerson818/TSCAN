import torch
import torch.nn as nn
from torch.nn import functional as F

import numbers

from losses.common import euclidean_dist
from losses.common import get_mask

from torch.autograd import Variable

# from enum import Enum
# class Difficulty(Enum):
#     Easy = 1        # A - P < A - N
#     SemiHard = 2    # min(A - N)
#     Hard = 3        # max(A - P), min(A - N)


# def _get_anchor_positive_triplet_mask(labels):
#     indices_equal = torch.eye(labels.size(0), device=labels.device).bool()
#     indices_not_equal = ~indices_equal
#     labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
#     return labels_equal & indices_not_equal


# def _get_anchor_negative_triplet_mask(labels):
#     return ~(labels.unsqueeze(0) == labels.unsqueeze(1))


# def _pairwise_distances(embeddings, squared=False, cosine=False):
#     dot_product = torch.matmul(embeddings, embeddings.t())
#     if cosine: # Cosine range is -1 to 1. 1 - similarity makes 0 be closest, 2 = furthest
#         norm = torch.norm(embeddings, dim=1, keepdim=True)
#         similarity = dot_product / torch.matmul(norm, norm.t())
#         return 1 - similarity

#     square_norm = torch.diag(dot_product)
#     distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
#     distances[distances < 0] = 0
#     if not squared:
#         mask = distances.eq(0).float()
#         distances = distances + mask * 1e-16
#         distances = (1.0 - mask) * torch.sqrt(distances)
#     return distances


# def _masked_minimum(data, mask, dim=1):
#     axis_maximums = data.max(dim, keepdim=True).values
#     masked_minimums = ((data - axis_maximums) * mask.float()).min(dim, keepdim=True).values + axis_maximums
#     return masked_minimums


# def _masked_maximum(data, mask, dim=1):
#     axis_minimums = data.min(dim, keepdim=True).values
#     masked_maximums = ((data - axis_minimums) * mask.float()).max(dim, keepdim=True).values + axis_minimums
#     return masked_maximums


# def batch_hard_triplet_loss(labels, embeddings, margin, squared=False, cosine=False):
#     pairwise_dist = _pairwise_distances(embeddings, squared=squared, cosine=cosine)
#     mask_anchor_positive = _get_anchor_positive_triplet_mask(labels).float()
#     anchor_positive_dist = mask_anchor_positive * pairwise_dist
#     hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)
#     mask_anchor_negative = _get_anchor_negative_triplet_mask(labels).float()
#     max_anchor_negative_dist, _ = pairwise_dist.max(1, keepdim=True)
#     anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)
#     hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True)
#     tl = hardest_positive_dist - hardest_negative_dist + margin
#     tl = F.relu(tl)
#     triplet_loss = tl.mean()
#     return triplet_loss


# class TripletLoss(torch.nn.Module):
#     def __init__(self, margin=0.2, cosine=False, difficulty=Difficulty.Hard):
#         super(TripletLoss, self).__init__()
#         self.margin = margin
#         self.cosine = cosine
#         self.difficulty = difficulty

#     # e.g. loss.change_parameter(difficulty=Difficulty.Hard)
#     def change_parameter(self, margin=None, cosine=None, difficulty=None):
#         self.margin = self.margin if margin is None else margin
#         self.cosine = self.cosine if cosine is None else cosine
#         self.difficulty = self.difficulty if difficulty is None else difficulty

#     def forward(self, embeddings, labels):
#         if self.difficulty == Difficulty.Hard:
#             return batch_hard_triplet_loss(labels, embeddings, self.margin, cosine=self.cosine)

#         adjacency_not = _get_anchor_negative_triplet_mask(labels)
#         batch_size = labels.size(0)

#         pdist_matrix = _pairwise_distances(embeddings, cosine=self.cosine)
#         pdist_matrix_tile = pdist_matrix.repeat(batch_size, 1)
#         mask = adjacency_not.repeat(batch_size, 1)

#         if self.difficulty == Difficulty.Easy:
#             mask = mask & torch.gt(pdist_matrix_tile, pdist_matrix.t().reshape(-1, 1))

#         mask_final = torch.gt(mask.float().sum(dim=1, keepdim=True), 0.0).reshape(batch_size, batch_size)
#         mask_final = mask_final.t()

#         adjacency_not = adjacency_not.float()
#         mask = mask.float()

#         negatives_outside = (
#             _masked_minimum(pdist_matrix_tile, mask)
#             .reshape(batch_size, batch_size)
#             .t()
#         )

#         negatives_inside = _masked_maximum(pdist_matrix, adjacency_not).repeat(1, batch_size)
#         semi_hard_negatives = torch.where(mask_final, negatives_outside, negatives_inside)

#         loss_mat = self.margin + pdist_matrix - semi_hard_negatives

#         mask_positives = _get_anchor_positive_triplet_mask(labels)
#         num_positives = torch.sum(mask_positives)
#         triplet_loss = torch.sum(torch.clamp(loss_mat * mask_positives.float(), min=0.0)) / (num_positives + 1e-8)
#         return triplet_loss


class TripletLoss(nn.Module):
    def __init__(self, margin=0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        inputs = nn.functional.normalize(inputs,p=2, dim=1, eps=1e-12)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        # print(dist)
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        # print(dist_ap)
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        return loss#, prec

# class TripletLoss(nn.Module):
#     """Triplet loss with hard positive/negative mining.
    
#     Reference:
#         Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    
#     Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.
    
#     Args:
#         margin (float, optional): margin for triplet. Default is 0.3.
#     """
    
#     def __init__(self, margin=0.3):
#         super(TripletLoss, self).__init__()
#         self.margin = margin
#         self.ranking_loss = nn.MarginRankingLoss(margin=margin)
 
#     def forward(self, inputs, targets):
#         """
#         Args:
#             inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
#             targets (torch.LongTensor): ground truth labels with shape (num_classes).
#         """
#         # inputs = F.normalize(inputs)
#         inputs = nn.functional.normalize(inputs,p=2, dim=1, eps=1e-12)
#         n = inputs.size(0)	# batch_size
        
#         # Compute pairwise distance, replace by the official when merged
#         dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
#         dist = dist + dist.t()
#         dist.addmm_(1, -2, inputs, inputs.t())
#         dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
#         # For each anchor, find the hardest positive and negative
#         mask = targets.expand(n, n).eq(targets.expand(n, n).t())
#         dist_ap, dist_an = [], []
#         for i in range(n):
#             dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
#             dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
#         dist_ap = torch.cat(dist_ap)
#         dist_an = torch.cat(dist_an)
        
#         # Compute ranking hinge loss
#         y = torch.ones_like(dist_an)
#         loss = self.ranking_loss(dist_an, dist_ap, y)
#         return loss

# class TripletLoss(nn.Module):
#     """
#     Batch hard triplet loss.
#     """
#     def __init__(self, margin):
#         super(TripletLoss, self).__init__()
#         if not (isinstance(margin, numbers.Real) or margin == 'soft'):
#             raise Exception('Invalid margin parameter for triplet loss.')
#         self.margin = margin

#     def forward(self, feature, label):
#         distance = euclidean_dist(feature, feature, squared=False)

#         positive_mask = get_mask(label, 'positive')
#         hardest_positive = (distance * positive_mask.float()).max(dim=1)[0]
        
#         negative_mask = get_mask(label, 'negative')
#         max_distance = distance.max(dim=1)[0]
#         not_negative_mask = ~(negative_mask.data)
#         negative_distance = distance + max_distance * (not_negative_mask.float())
#         hardest_negative = negative_distance.min(dim=1)[0]

#         diff = hardest_positive - hardest_negative
#         if isinstance(self.margin, numbers.Real):
#             tri_loss = (self.margin + diff).clamp(min=0).mean()
#         else:
#             tri_loss = F.softplus(diff).mean()
        
#         return tri_loss
