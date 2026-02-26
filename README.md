# 三维栅格地图路径规划深度学习项目

## 项目概述
本项目使用深度学习方法（U-Net、Transformer）实现三维栅格地图的最短路径规划，将路径规划问题转化为三维图像分割问题。

## 项目结构
```
py05/
├── data/                    # 数据相关
│   ├── astar.py             # A*算法实现
│   ├── generate_data.py     # 数据生成脚本
│   ├── dataset.py           # 数据集加载器
│   └── data/                # 生成的数据集
├── models/                  # 模型定义
│   ├── unet3d.py            # 3D U-Net模型
│   └── transformer.py       # 3D Transformer模型
├── train.py                 # 训练脚本
├── test.py                  # 测试脚本
├── evaluate.py              # 评估脚本
├── visualize.py             # 可视化工具
├── utils.py                 # 工具函数
├── requirements.txt         # 依赖文件
├── 实现方案.md              # 详细实现方案
└── 项目交接文档.md          # 项目交接文档
```

## 已完成工作

### 1. 数据部分
- ✅ 三维A*和Dijkstra算法实现（astar.py）
- ✅ 数据集生成脚本（generate_data.py）
- ✅ PyTorch数据集加载器（dataset.py）
- ✅ 成功生成3000组训练数据（2400训练+300验证+300测试）

### 2. 模型部分
- ✅ 3D U-Net模型（unet3d.py）
- ✅ 3D Transformer模型（transformer.py）
- ✅ 工具函数（utils.py）：损失函数、学习率调度器、评估指标等

### 3. 训练和测试
- ✅ 训练脚本（train.py）：支持两种模型的训练
- ✅ 测试脚本（test.py）：在测试集上评估模型性能
- ✅ 评估脚本（evaluate.py）：计算成功率、路径长度、推理时间等指标

### 4. 可视化
- ✅ 可视化工具（visualize.py）：三维路径可视化、训练曲线绘制、结果对比图

## 环境配置

### 依赖安装
```bash
# 激活虚拟环境
.envcriptsctivate.ps1

# 安装依赖
pip install -r requirements.txt
```

### PyTorch环境问题解决
如果遇到PyTorch的DLL加载问题（如`[WinError 1114] 动态链接库(DLL)初始化例程失败`），可以尝试以下解决方法：

1. **重新安装PyTorch**：
   ```bash
   pip uninstall torch
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

2. **检查Python版本**：确保使用的Python版本与PyTorch兼容（推荐Python 3.8-3.10）

3. **更新系统DLL**：安装或更新Windows的Microsoft Visual C++ Redistributable

## 使用说明

### 1. 生成数据集
```bash
python data/generate_data.py
```

### 2. 训练模型
```bash
# 训练U-Net模型
python train.py --model_type unet3d

# 训练Transformer模型
python train.py --model_type transformer
```

### 3. 测试模型
```bash
# 测试U-Net模型
python test.py --model_type unet3d --model_path ./models/unet3d/best_model_epoch50.pth

# 测试Transformer模型
python test.py --model_type transformer --model_path ./models/transformer/best_model_epoch50.pth
```

### 4. 评估模型
```bash
# 评估U-Net模型
python evaluate.py --model_type unet3d --model_path ./models/unet3d/best_model_epoch50.pth

# 评估Transformer模型
python evaluate.py --model_type transformer --model_path ./models/transformer/best_model_epoch50.pth
```

### 5. 可视化结果
```bash
# 使用可视化工具
python visualize.py
```

## 技术参数

### 地图参数
- 地图尺寸：16×16×16
- 障碍物密度：10%-30%
- 数据集规模：3000组（训练2400 + 验证300 + 测试300）

### 模型参数
- 输入通道：4（障碍物 + 起点 + 终点 + 可行区域）
- 输出通道：1（路径概率）
- Batch Size：4
- 学习率：0.001
- 训练轮次：50-100轮

### 损失函数
使用组合损失函数：
```
loss = 0.5 * BCELoss + 0.5 * DiceLoss
```

## 注意事项

1. **显存/内存不足**：减小batch_size或地图尺寸
2. **训练慢**：先用少量数据测试代码正确性
3. **路径不连续**：可添加连通性损失或后处理
4. **过拟合**：增加数据增强、使用Dropout

## 预期成果

1. 完整代码：数据生成、模型、训练、测试
2. 数据集：3000组三维栅格地图及路径标签
3. 训练模型：U-Net和Transformer模型
4. 实验报告：两种模型的对比分析
5. 毕业论文