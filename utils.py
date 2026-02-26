import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """Dice损失函数，用于处理不平衡数据"""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        # 将输入和目标展平
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # 计算交集
        intersection = (inputs * targets).sum()
        
        # 计算Dice系数
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        # 返回Dice损失（1 - Dice系数）
        return 1 - dice

class CombinedLoss(nn.Module):
    """组合损失函数：BCELoss + DiceLoss"""
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-6):
        super().__init__()
        self.bce_loss = nn.BCELoss()
        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
    
    def forward(self, inputs, targets):
        """前向传播"""
        # 确保目标类型与输入一致
        targets = targets.float()
        
        # 确保目标形状与输入一致（添加通道维度）
        if inputs.dim() == 5 and targets.dim() == 4:
            targets = targets.unsqueeze(1)  # 在通道维度添加一个维度
        
        # 计算各部分损失
        bce = self.bce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        
        # 组合损失
        loss = self.bce_weight * bce + self.dice_weight * dice
        
        return loss

# 学习率调度器
class LearningRateScheduler:
    """简单的学习率调度器"""
    def __init__(self, optimizer, initial_lr=0.001, lr_decay=0.1, decay_epochs=20):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.lr_decay = lr_decay
        self.decay_epochs = decay_epochs
    
    def step(self, epoch):
        """根据当前 epoch 更新学习率"""
        lr = self.initial_lr * (self.lr_decay ** (epoch // self.decay_epochs))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

# 评估指标
def calculate_metrics(preds, targets):
    """计算评估指标：accuracy, precision, recall, f1"""
    # 二值化预测结果
    preds = (preds > 0.5).float()
    targets = targets.float()
    
    # 展平张量
    preds = preds.view(-1)
    targets = targets.view(-1)
    
    # 计算TP, FP, TN, FN
    TP = (preds * targets).sum().item()
    FP = (preds * (1 - targets)).sum().item()
    TN = ((1 - preds) * (1 - targets)).sum().item()
    FN = ((1 - preds) * targets).sum().item()
    
    # 计算指标
    accuracy = (TP + TN) / (TP + FP + TN + FN + 1e-6)
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# 保存和加载模型
def save_model(model, optimizer, epoch, loss, file_path):
    """保存模型"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, file_path)
    print(f"模型已保存至: {file_path}")

def load_model(model, optimizer, file_path):
    """加载模型"""
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"模型已加载自: {file_path}")
    return epoch, loss