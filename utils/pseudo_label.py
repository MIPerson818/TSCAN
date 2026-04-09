import torch
import torch.nn.functional as F

def generate_pseudo_labels(teacher_feat, distance_threshold=0.5):
    """
    Args:
        teacher_feat: 教师对目标域弱增强图像的特征 (B, feature_dim)
        distance_threshold: 判定为同一id的距离阈值
    Returns:
        pseudo_labels: 伪标签 (B,)
    """
    # 计算特征间的余弦距离（1 - 余弦相似度）
    cos_sim = F.cosine_similarity(teacher_feat.unsqueeze(1), teacher_feat.unsqueeze(0), dim=2)  # (B,B)
    distance = 1 - cos_sim  # 余弦距离：0（相同）~2（相反）

    # 初始化伪标签（从0开始）
    pseudo_labels = torch.zeros(teacher_feat.shape[0], dtype=torch.long, device=teacher_feat.device)
    current_id = 0

    for i in range(teacher_feat.shape[0]):
        if pseudo_labels[i] != 0:  # 已分配标签
            continue
        # 找到与i距离小于阈值的样本（同一id）
        same_id_mask = (distance[i] < distance_threshold)
        pseudo_labels[same_id_mask] = current_id
        current_id += 1

    return pseudo_labels