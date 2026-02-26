import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.astar import astar_3d, dijkstra_3d
from typing import Tuple, Optional
import json


def generate_map(size: int = 16, obstacle_ratio: float = 0.2) -> np.ndarray:
    grid = np.zeros((size, size, size), dtype=np.float32)
    obstacle_count = int(size ** 3 * obstacle_ratio)
    
    indices = np.random.choice(size ** 3, obstacle_count, replace=False)
    for idx in indices:
        x, y, z = np.unravel_index(idx, (size, size, size))
        grid[x, y, z] = 1
    
    return grid


def get_free_cells(grid: np.ndarray) -> np.ndarray:
    return np.argwhere(grid == 0)


def select_start_end(free_cells: np.ndarray, min_distance: int = 5) -> Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
    if len(free_cells) < 2:
        return None
    
    for _ in range(100):
        idx = np.random.choice(len(free_cells), 2, replace=False)
        start = tuple(free_cells[idx[0]])
        end = tuple(free_cells[idx[1]])
        
        dist = abs(start[0] - end[0]) + abs(start[1] - end[1]) + abs(start[2] - end[2])
        if dist >= min_distance:
            return start, end
    
    return None


def create_sample(size: int = 16, min_obstacle: float = 0.1, max_obstacle: float = 0.3, 
                  min_distance: int = 5, algorithm: str = 'astar') -> Optional[Tuple[np.ndarray, np.ndarray, dict]]:
    obstacle_ratio = np.random.uniform(min_obstacle, max_obstacle)
    grid = generate_map(size, obstacle_ratio)
    
    free_cells = get_free_cells(grid)
    if len(free_cells) < 2:
        return None
    
    result = select_start_end(free_cells, min_distance)
    if result is None:
        return None
    
    start, end = result
    
    if algorithm == 'astar':
        path = astar_3d(grid, start, end)
    else:
        path = dijkstra_3d(grid, start, end)
    
    if path is None:
        return None
    
    input_data = np.zeros((4, size, size, size), dtype=np.float32)
    input_data[0] = grid
    input_data[1][start] = 1
    input_data[2][end] = 1
    input_data[3] = 1 - grid
    
    label = np.zeros((size, size, size), dtype=np.float32)
    for p in path:
        label[p] = 1
    
    # 将numpy类型转换为Python原生类型，以便JSON序列化
    info = {
        'start': tuple(int(coord) for coord in start),
        'end': tuple(int(coord) for coord in end),
        'path_length': len(path),
        'obstacle_ratio': float(obstacle_ratio)
    }
    
    return input_data, label, info


def generate_dataset(output_dir: str, num_samples: int = 3000, size: int = 16,
                     train_ratio: float = 0.8, val_ratio: float = 0.1):
    os.makedirs(output_dir, exist_ok=True)
    
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    for d in [train_dir, val_dir, test_dir]:
        os.makedirs(d, exist_ok=True)
    
    num_train = int(num_samples * train_ratio)
    num_val = int(num_samples * val_ratio)
    num_test = num_samples - num_train - num_val
    
    splits = [
        ('train', train_dir, num_train),
        ('val', val_dir, num_val),
        ('test', test_dir, num_test)
    ]
    
    all_info = {}
    
    for split_name, split_dir, count in splits:
        print(f"生成 {split_name} 数据集，数量: {count}")
        samples_generated = 0
        attempts = 0
        max_attempts = count * 10
        
        while samples_generated < count and attempts < max_attempts:
            attempts += 1
            result = create_sample(size)
            
            if result is None:
                continue
            
            input_data, label, info = result
            
            sample_idx = samples_generated
            np.save(os.path.join(split_dir, f'input_{sample_idx}.npy'), input_data)
            np.save(os.path.join(split_dir, f'label_{sample_idx}.npy'), label)
            
            all_info[f'{split_name}_{sample_idx}'] = info
            samples_generated += 1
            
            if samples_generated % 100 == 0:
                print(f"  已生成 {samples_generated}/{count}")
        
        print(f"  {split_name} 完成，共生成 {samples_generated} 个样本")
    
    with open(os.path.join(output_dir, 'dataset_info.json'), 'w') as f:
        json.dump(all_info, f, indent=2)
    
    print(f"数据集生成完成，保存至: {output_dir}")


if __name__ == "__main__":
    generate_dataset(
        output_dir='./data/dataset',
        num_samples=3000,
        size=16
    )