import os
import torch
import numpy as np
import json

# 导入自定义模块
from data.dataset import get_dataloaders
from models.unet3d import UNet3D
from models.transformer import Transformer3D
from models.hybrid import HybridUNetTransformer
from utils import calculate_metrics

def test_single_model(model_type, model_path, config):
    """测试单个模型"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n===== 测试 {model_type} 模型 =====")
    print(f"模型路径: {model_path}")
    
    # 创建数据加载器（只使用测试集）
    _, _, test_loader = get_dataloaders(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    # 创建模型
    if model_type == 'unet3d':
        model = UNet3D(in_channels=4, out_channels=1)
    elif model_type == 'transformer':
        model = Transformer3D(in_channels=4, out_channels=1, grid_size=config['grid_size'])
    elif model_type == 'hybrid':
        model = HybridUNetTransformer(in_channels=4, out_channels=1, grid_size=config['grid_size'])
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    model = model.to(device)
    
    # 加载模型权重
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"模型已加载，训练轮次: {checkpoint['epoch']}")
    
    # 设置模型为评估模式
    model.eval()
    
    # 初始化测试指标
    test_metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
    
    # 保存预测结果
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        from tqdm import tqdm
        for inputs, targets in tqdm(test_loader, desc=f"测试 {model_type}"):
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
    
    # 计算平均测试指标
    for key in test_metrics:
        test_metrics[key] = test_metrics[key] / len(test_loader.dataset)
    
    # 输出测试结果
    print(f"测试集大小: {len(test_loader.dataset)}")
    print(f"测试指标: {test_metrics}")
    
    # 将预测结果保存为numpy文件
    results_dir = os.path.join(config['output_dir'], model_type)
    os.makedirs(results_dir, exist_ok=True)
    
    np.save(os.path.join(results_dir, 'test_preds.npy'), np.concatenate(all_preds))
    np.save(os.path.join(results_dir, 'test_targets.npy'), np.concatenate(all_targets))
    print(f"预测结果已保存至: {results_dir}")
    
    # 将测试结果保存到文件
    with open(os.path.join(results_dir, 'test_results.txt'), 'w') as f:
        f.write(f"===== {model_type} 模型测试结果 =====\n")
        f.write(f"模型路径: {model_path}\n")
        f.write(f"测试集大小: {len(test_loader.dataset)}\n")
        for key, value in test_metrics.items():
            f.write(f"{key}: {value:.4f}\n")
    
    print(f"测试结果已保存至: {os.path.join(results_dir, 'test_results.txt')}")
    
    return test_metrics

def test_all_models():
    """测试所有模型"""
    # 配置参数
    config = {
        'data_dir': './data/data/dataset',  # 数据集路径
        'output_dir': './results',  # 结果保存目录
        'grid_size': (16, 16, 16),  # 地图尺寸
        'batch_size': 4,  # 批大小
        'num_workers': 0,  # 数据加载线程数
    }
    
    # 模型配置
    models = [
        {
            'type': 'unet3d',
            'path': './models/unet3d/final_model_epoch50.pth'
        },
        {
            'type': 'transformer',
            'path': './models/transformer/final_model_epoch50.pth'
        },
        {
            'type': 'hybrid',
            'path': './models/hybrid/final_model_epoch50.pth'
        }
    ]
    
    print("开始批量测试所有模型...")
    print(f"测试配置: {config}")
    
    # 测试所有模型
    all_results = {}
    for model_config in models:
        model_type = model_config['type']
        model_path = model_config['path']
        
        if os.path.exists(model_path):
            try:
                results = test_single_model(model_type, model_path, config)
                all_results[model_type] = results
            except Exception as e:
                print(f"测试 {model_type} 模型失败: {e}")
                all_results[model_type] = {'error': str(e)}
        else:
            print(f"跳过 {model_type} 模型: 模型文件不存在 {model_path}")
            all_results[model_type] = {'error': f'模型文件不存在: {model_path}'}
    
    # 保存所有模型的测试结果对比
    comparison_path = os.path.join(config['output_dir'], 'models_comparison.json')
    with open(comparison_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n===== 所有模型测试完成 =====")
    print(f"测试结果对比已保存至: {comparison_path}")
    
    # 打印对比结果
    print("\n模型性能对比:")
    print("-" * 60)
    print(f"{'模型类型':<15} {'准确率':<10} {'精确率':<10} {'召回率':<10} {'F1分数':<10}")
    print("-" * 60)
    for model_type, results in all_results.items():
        if 'error' not in results:
            print(f"{model_type:<15} {results['accuracy']:<10.4f} {results['precision']:<10.4f} {results['recall']:<10.4f} {results['f1']:<10.4f}")
        else:
            print(f"{model_type:<15} {'错误':<10} {'错误':<10} {'错误':<10} {'错误':<10}")
    print("-" * 60)

if __name__ == "__main__":
    test_all_models()