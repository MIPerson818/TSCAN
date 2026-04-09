import torchvision.transforms as transforms

def get_weak_augmentation(size=128):
    """弱增强：随机水平翻转+裁剪（不影响特征本质）"""
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(size, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_strong_augmentation(size=128):
    """强增强：颜色抖动+灰度化+高斯模糊（影响域分布）"""
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(size, padding=4),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])