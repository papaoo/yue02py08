import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json

class Visualizer:
    """可视化工具类"""
    
    @staticmethod
    def plot_training_curves(train_losses, val_losses, metrics, save_path=None):
        """绘制训练和验证曲线"""
        plt.figure(figsize=(12, 6))
        
        # 绘制损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='训练损失')
        plt.plot(val_losses, label='验证损失')
        plt.title('训练和验证损失曲线')
        plt.xlabel('Epoch')
        plt.ylabel('损失值')
        plt.legend()
        plt.grid(True)
        
        # 绘制指标曲线
        plt.subplot(1, 2, 2)
        for metric_name, values in metrics.items():
            plt.plot(values, label=metric_name)
        plt.title('训练指标曲线')
        plt.xlabel('Epoch')
        plt.ylabel('指标值')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"训练曲线已保存至: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def visualize_3d_path(grid, path, start, end, title='三维路径可视化', save_path=None):
        """可视化三维栅格地图和路径"""
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 获取障碍物坐标
        obstacle_coords = np.argwhere(grid == 1)
        
        # 获取路径坐标
        path_coords = np.array(path)
        
        # 绘制障碍物
        if obstacle_coords.size > 0:
            ax.scatter(
                obstacle_coords[:, 0], obstacle_coords[:, 1], obstacle_coords[:, 2],
                c='gray', marker='s', s=100, alpha=0.5, label='障碍物'
            )
        
        # 绘制路径
        if path_coords.size > 0:
            ax.plot(
                path_coords[:, 0], path_coords[:, 1], path_coords[:, 2],
                c='blue', marker='o', linestyle='-', linewidth=2, markersize=5, label='预测路径'
            )
        
        # 绘制起点和终点
        ax.scatter(start[0], start[1], start[2], c='green', marker='^', s=200, label='起点')
        ax.scatter(end[0], end[1], end[2], c='red', marker='v', s=200, label='终点')
        
        # 设置坐标轴标签
        ax.set_xlabel('X轴')
        ax.set_ylabel('Y轴')
        ax.set_zlabel('Z轴')
        
        # 设置标题
        ax.set_title(title)
        
        # 设置坐标轴范围
        size = grid.shape[0]
        ax.set_xlim(0, size-1)
        ax.set_ylim(0, size-1)
        ax.set_zlim(0, size-1)
        
        # 添加图例
        ax.legend()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"三维路径可视化已保存至: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def compare_paths(grid, pred_path, true_path, start, end, title='路径对比', save_path=None):
        """对比真实路径和预测路径"""
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 获取障碍物坐标
        obstacle_coords = np.argwhere(grid == 1)
        
        # 将路径转换为numpy数组
        pred_path_coords = np.array(pred_path)
        true_path_coords = np.array(true_path)
        
        # 绘制障碍物
        if obstacle_coords.size > 0:
            ax.scatter(
                obstacle_coords[:, 0], obstacle_coords[:, 1], obstacle_coords[:, 2],
                c='gray', marker='s', s=100, alpha=0.5, label='障碍物'
            )
        
        # 绘制真实路径
        if true_path_coords.size > 0:
            ax.plot(
                true_path_coords[:, 0], true_path_coords[:, 1], true_path_coords[:, 2],
                c='green', marker='o', linestyle='-', linewidth=2, markersize=5, label='真实路径'
            )
        
        # 绘制预测路径
        if pred_path_coords.size > 0:
            ax.plot(
                pred_path_coords[:, 0], pred_path_coords[:, 1], pred_path_coords[:, 2],
                c='red', marker='x', linestyle='-', linewidth=2, markersize=5, label='预测路径'
            )
        
        # 绘制起点和终点
        ax.scatter(start[0], start[1], start[2], c='blue', marker='^', s=200, label='起点')
        ax.scatter(end[0], end[1], end[2], c='purple', marker='v', s=200, label='终点')
        
        # 设置坐标轴标签
        ax.set_xlabel('X轴')
        ax.set_ylabel('Y轴')
        ax.set_zlabel('Z轴')
        
        # 设置标题
        ax.set_title(title)
        
        # 设置坐标轴范围
        size = grid.shape[0]
        ax.set_xlim(0, size-1)
        ax.set_ylim(0, size-1)
        ax.set_zlim(0, size-1)
        
        # 添加图例
        ax.legend()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"路径对比图已保存至: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def visualize_slice(input_data, slice_idx, axis=0, title='切片可视化', save_path=None):
        """可视化三维数据的切片"""
        # 选择要显示的切片
        if axis == 0:
            obstacle_slice = input_data[0, slice_idx, :, :]
            start_slice = input_data[1, slice_idx, :, :]
            end_slice = input_data[2, slice_idx, :, :]
            free_slice = input_data[3, slice_idx, :, :]
        elif axis == 1:
            obstacle_slice = input_data[0, :, slice_idx, :]
            start_slice = input_data[1, :, slice_idx, :]
            end_slice = input_data[2, :, slice_idx, :]
            free_slice = input_data[3, :, slice_idx, :]
        elif axis == 2:
            obstacle_slice = input_data[0, :, :, slice_idx]
            start_slice = input_data[1, :, :, slice_idx]
            end_slice = input_data[2, :, :, slice_idx]
            free_slice = input_data[3, :, :, slice_idx]
        else:
            raise ValueError(f"无效的轴: {axis}")
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        
        # 绘制障碍物切片
        im0 = axes[0, 0].imshow(obstacle_slice, cmap='gray')
        axes[0, 0].set_title('障碍物通道')
        plt.colorbar(im0, ax=axes[0, 0])
        
        # 绘制起点切片
        im1 = axes[0, 1].imshow(start_slice, cmap='viridis')
        axes[0, 1].set_title('起点通道')
        plt.colorbar(im1, ax=axes[0, 1])
        
        # 绘制终点切片
        im2 = axes[1, 0].imshow(end_slice, cmap='viridis')
        axes[1, 0].set_title('终点通道')
        plt.colorbar(im2, ax=axes[1, 0])
        
        # 绘制可行区域切片
        im3 = axes[1, 1].imshow(free_slice, cmap='gray')
        axes[1, 1].set_title('可行区域通道')
        plt.colorbar(im3, ax=axes[1, 1])
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"切片可视化已保存至: {save_path}")
        else:
            plt.show()
        
        plt.close()

# 示例用法
if __name__ == "__main__":
    # 加载数据集信息
    info_path = './data/data/dataset/dataset_info.json'
    with open(info_path, 'r') as f:
        dataset_info = json.load(f)
    
    # 加载一个样本
    sample_idx = 0
    input_path = f'./data/data/dataset/test/input_{sample_idx}.npy'
    label_path = f'./data/data/dataset/test/label_{sample_idx}.npy'
    
    input_data = np.load(input_path)
    label_data = np.load(label_path)
    
    # 获取样本信息
    info_key = f'test_{sample_idx}'
    info = dataset_info[info_key]
    start = tuple(info['start'])
    end = tuple(info['end'])
    
    # 提取障碍物地图
    obstacle_map = input_data[0]
    
    # 提取真实路径
    true_path = np.argwhere(label_data == 1).tolist()
    
    # 可视化切片
    Visualizer.visualize_slice(input_data, slice_idx=8, axis=0, title=f'样本 {sample_idx} - X=8 切片', 
                              save_path=f'./results/visualization/sample_{sample_idx}_slice.png')
    
    # 可视化真实路径
    Visualizer.visualize_3d_path(obstacle_map, true_path, start, end, 
                                title=f'样本 {sample_idx} - 真实路径',
                                save_path=f'./results/visualization/sample_{sample_idx}_true_path.png')