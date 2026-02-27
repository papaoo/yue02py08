import os
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# 导入自定义模块
from data.dataset import get_dataloaders
from models.hybrid import HybridUNetTransformer
from utils import CombinedLoss, LearningRateScheduler, save_model, calculate_metrics

def train_hybrid():
    """训练U-Net + Transformer混合模型"""
    # 配置参数
    config = {
        'model_type': 'hybrid',
        'data_dir': './data/data/dataset',
        'output_dir': './models',
        'grid_size': (16, 16, 16),
        'batch_size': 4,
        'learning_rate': 0.001,
        'num_epochs': 20,
        'num_workers': 0,
    }
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建数据加载器
    train_loader, val_loader, _ = get_dataloaders(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    # 创建模型
    model = HybridUNetTransformer(in_channels=4, out_channels=1, grid_size=config['grid_size'])
    model = model.to(device)
    
    # 创建损失函数和优化器
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # 学习率调度器
    scheduler = LearningRateScheduler(optimizer, initial_lr=config['learning_rate'])
    
    # 创建模型保存目录
    model_dir = os.path.join(config['output_dir'], config['model_type'])
    os.makedirs(model_dir, exist_ok=True)
    
    # 最佳验证损失
    best_val_loss = float('inf')
    
    # 训练循环
    for epoch in range(config['num_epochs']):
        print(f"\n===== Epoch {epoch+1}/{config['num_epochs']} =====")
        
        # 更新学习率
        current_lr = scheduler.step(epoch)
        print(f"当前学习率: {current_lr:.6f}")
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
        
        with tqdm(total=len(train_loader), desc="训练") as pbar:
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # 前向传播
                outputs = model(inputs)
                
                # 计算损失
                loss = criterion(outputs, targets)
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 更新训练损失
                train_loss += loss.item() * inputs.size(0)
                
                # 计算指标
                metrics = calculate_metrics(outputs, targets)
                for key in train_metrics:
                    train_metrics[key] += metrics[key] * inputs.size(0)
                
                pbar.update(1)
        
        # 计算平均训练损失和指标
        train_loss = train_loss / len(train_loader.dataset)
        for key in train_metrics:
            train_metrics[key] = train_metrics[key] / len(train_loader.dataset)
        
        print(f"训练损失: {train_loss:.4f}")
        print(f"训练指标: {train_metrics}")
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
        
        with torch.no_grad():
            with tqdm(total=len(val_loader), desc="验证") as pbar:
                for inputs, targets in val_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    
                    # 前向传播
                    outputs = model(inputs)
                    
                    # 计算损失
                    loss = criterion(outputs, targets)
                    
                    # 更新验证损失
                    val_loss += loss.item() * inputs.size(0)
                    
                    # 计算指标
                    metrics = calculate_metrics(outputs, targets)
                    for key in val_metrics:
                        val_metrics[key] += metrics[key] * inputs.size(0)
                    
                    pbar.update(1)
        
        # 计算平均验证损失和指标
        val_loss = val_loss / len(val_loader.dataset)
        for key in val_metrics:
            val_metrics[key] = val_metrics[key] / len(val_loader.dataset)
        
        print(f"验证损失: {val_loss:.4f}")
        print(f"验证指标: {val_metrics}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(model_dir, f"best_model_epoch{epoch+1}.pth")
            save_model(model, optimizer, epoch+1, val_loss, best_model_path)
    
    # 保存最后一个 epoch 的模型
    final_model_path = os.path.join(model_dir, f"final_model_epoch{config['num_epochs']}.pth")
    save_model(model, optimizer, config['num_epochs'], val_loss, final_model_path)
    
    print(f"\n训练完成！")
    print(f"最佳验证损失: {best_val_loss:.4f}")

if __name__ == "__main__":
    print("开始训练U-Net + Transformer混合模型...")
    train_hybrid()