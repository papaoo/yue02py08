import os
import torch
import numpy as np
from tqdm import tqdm

# 导入自定义模块
from data.dataset import get_dataloaders
from models.unet3d import UNet3D
from models.transformer import Transformer3D
from utils import calculate_metrics

def test_model(config):
    """测试模型"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建数据加载器（只使用测试集）
    _, _, test_loader = get_dataloaders(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    # 创建模型
    if config['model_type'] == 'unet3d':
        model = UNet3D(in_channels=4, out_channels=1)
    elif config['model_type'] == 'transformer':
        model = Transformer3D(in_channels=4, out_channels=1, grid_size=config['grid_size'])
    else:
        raise ValueError(f"不支持的模型类型: {config['model_type']}")
    
    model = model.to(device)
    
    # 加载模型权重
    if not os.path.exists(config['model_path']):
        raise FileNotFoundError(f"模型文件不存在: {config['model_path']}")
    
    checkpoint = torch.load(config['model_path'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"模型已加载自: {config['model_path']}")
    
    # 设置模型为评估模式
    model.eval()
    
    # 初始化测试指标
    test_loss = 0.0
    test_metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
    
    # 保存预测结果
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        with tqdm(total=len(test_loader), desc="测试") as pbar:
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # 前向传播
                outputs = model(inputs)
                
                # 保存预测结果
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                
                # 计算指标
                metrics = calculate_metrics(outputs, targets)
                for key in test_metrics:
                    test_metrics[key] += metrics[key] * inputs.size(0)
                
                pbar.update(1)
    
    # 计算平均测试指标
    for key in test_metrics:
        test_metrics[key] = test_metrics[key] / len(test_loader.dataset)
    
    # 输出测试结果
    print("\n===== 测试结果 =====")
    print(f"测试集大小: {len(test_loader.dataset)}")
    print(f"测试指标: {test_metrics}")
    
    # 将预测结果保存为numpy文件
    if config['save_results']:
        results_dir = os.path.join(config['output_dir'], config['model_type'])
        os.makedirs(results_dir, exist_ok=True)
        
        np.save(os.path.join(results_dir, 'test_preds.npy'), np.concatenate(all_preds))
        np.save(os.path.join(results_dir, 'test_targets.npy'), np.concatenate(all_targets))
        print(f"\n预测结果已保存至: {results_dir}")
    
    return test_metrics

if __name__ == "__main__":
    # 配置参数
    config = {
        'model_type': 'unet3d',  # 可选: 'unet3d' 或 'transformer'
        'data_dir': './data/data/dataset',  # 数据集路径
        'model_path': './models/unet3d/final_model_epoch50.pth',  # 模型路径
        'output_dir': './results',  # 结果保存目录
        'grid_size': (16, 16, 16),  # 地图尺寸
        'batch_size': 4,  # 批大小
        'num_workers': 0,  # 数据加载线程数
        'save_results': True,  # 是否保存预测结果
    }
    
    print("测试配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 开始测试
    test_metrics = test_model(config)
    
    # 将测试结果保存到文件
    results_dir = os.path.join(config['output_dir'], config['model_type'])
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, 'test_results.txt'), 'w') as f:
        f.write("===== 测试结果 =====\n")
        f.write(f"测试集大小: {len(get_dataloaders(config['data_dir'], 1, 0)[2].dataset)}\n")
        for key, value in test_metrics.items():
            f.write(f"{key}: {value:.4f}\n")
    
    print(f"\n测试结果已保存至: {os.path.join(results_dir, 'test_results.txt')}")