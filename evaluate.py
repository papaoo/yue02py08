import os
import time
import torch
import numpy as np
from tqdm import tqdm
from scipy.ndimage import label

# 导入自定义模块
from data.dataset import get_dataloaders
from models.unet3d import UNet3D
from models.transformer import Transformer3D

def find_longest_path(pred_map, start, end):
    """从预测概率图中找到最长的连通路径"""
    # 将概率图转换为二值图
    binary_map = (pred_map > 0.5).astype(int)
    
    # 检查起点和终点是否在二值图中
    if binary_map[start] == 0 or binary_map[end] == 0:
        return None
    
    # 使用连通区域标记
    labeled_map, num_features = label(binary_map)
    
    # 检查起点和终点是否在同一个连通区域
    start_label = labeled_map[start]
    end_label = labeled_map[end]
    
    if start_label == 0 or end_label == 0 or start_label != end_label:
        return None
    
    # 获取连通区域
    connected_region = np.argwhere(labeled_map == start_label)
    
    # 简单返回连通区域（实际应用中可能需要更复杂的路径提取算法）
    return connected_region.tolist()

def calculate_path_smoothness(path):
    """计算路径平滑度（相邻节点方向变化的次数）"""
    if len(path) < 3:
        return 0.0
    
    changes = 0
    for i in range(1, len(path)-1):
        # 计算相邻三个点的方向
        dir1 = np.array(path[i]) - np.array(path[i-1])
        dir2 = np.array(path[i+1]) - np.array(path[i])
        
        # 如果方向变化，计数加1
        if not np.array_equal(dir1, dir2):
            changes += 1
    
    # 平滑度为1 - 变化次数/(总点数-2)
    return 1.0 - (changes / (len(path) - 2))

def evaluate_model(config):
    """评估模型性能"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建数据加载器（只使用测试集）
    _, _, test_loader = get_dataloaders(
        data_dir=config['data_dir'],
        batch_size=1,  # 批大小为1，方便单独评估每个样本
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
    
    # 初始化评估指标
    total_samples = 0
    successful_samples = 0
    total_path_length = 0
    total_smoothness = 0
    total_inference_time = 0.0
    
    # 加载数据集信息
    info_path = os.path.join(config['data_dir'], 'dataset_info.json')
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"数据集信息文件不存在: {info_path}")
    
    import json
    with open(info_path, 'r') as f:
        dataset_info = json.load(f)
    
    with torch.no_grad():
        with tqdm(total=len(test_loader), desc="评估") as pbar:
            for idx, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # 记录推理时间
                start_time = time.time()
                outputs = model(inputs)
                end_time = time.time()
                inference_time = end_time - start_time
                total_inference_time += inference_time
                
                # 获取样本信息
                info_key = f'test_{idx}'
                if info_key not in dataset_info:
                    print(f"警告: 样本 {info_key} 的信息不存在")
                    pbar.update(1)
                    continue
                
                info = dataset_info[info_key]
                start = tuple(info['start'])
                end = tuple(info['end'])
                
                # 提取预测结果
                pred_map = outputs[0, 0].cpu().numpy()  # 形状: (16, 16, 16)
                
                # 查找路径
                path = find_longest_path(pred_map, start, end)
                
                if path is not None and len(path) > 0:
                    successful_samples += 1
                    total_path_length += len(path)
                    
                    # 计算路径平滑度
                    smoothness = calculate_path_smoothness(path)
                    total_smoothness += smoothness
                
                total_samples += 1
                pbar.update(1)
    
    # 计算平均指标
    success_rate = successful_samples / total_samples
    avg_path_length = total_path_length / successful_samples if successful_samples > 0 else 0
    avg_smoothness = total_smoothness / successful_samples if successful_samples > 0 else 0
    avg_inference_time = total_inference_time / total_samples
    
    # 输出评估结果
    print("\n===== 评估结果 =====")
    print(f"测试集大小: {total_samples}")
    print(f"成功路径数: {successful_samples}")
    print(f"成功率: {success_rate:.4f} ({successful_samples}/{total_samples})")
    print(f"平均路径长度: {avg_path_length:.4f}")
    print(f"平均路径平滑度: {avg_smoothness:.4f}")
    print(f"平均推理时间: {avg_inference_time*1000:.4f} ms")
    
    # 将评估结果保存到文件
    results_dir = os.path.join(config['output_dir'], config['model_type'])
    os.makedirs(results_dir, exist_ok=True)
    
    evaluation_results = {
        'test_set_size': total_samples,
        'successful_samples': successful_samples,
        'success_rate': success_rate,
        'avg_path_length': avg_path_length,
        'avg_smoothness': avg_smoothness,
        'avg_inference_time_ms': avg_inference_time * 1000,
        'model_path': config['model_path'],
        'evaluation_time': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(results_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    print(f"\n评估结果已保存至: {os.path.join(results_dir, 'evaluation_results.json')}")
    
    return evaluation_results

if __name__ == "__main__":
    # 配置参数
    config = {
        'model_type': 'unet3d',  # 可选: 'unet3d' 或 'transformer'
        'data_dir': './data/data/dataset',  # 数据集路径
        'model_path': './models/unet3d/final_model_epoch50.pth',  # 模型路径
        'output_dir': './results',  # 结果保存目录
        'grid_size': (16, 16, 16),  # 地图尺寸
        'num_workers': 0,  # 数据加载线程数
    }
    
    print("评估配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 开始评估
    evaluate_model(config)