import torch
import torch.nn as nn
import torch.nn.functional as F

from models.unet3d import UNet3D
from models.transformer import PositionalEncoding3D

class HybridUNetTransformer(nn.Module):
    """U-Net + Transformer混合模型"""
    def __init__(self, in_channels=4, out_channels=1, grid_size=(16, 16, 16), 
                 unet_features=[32, 64, 128, 256], 
                 transformer_dim=256, num_heads=4, num_transformer_layers=2, dropout=0.1):
        super().__init__()
        self.grid_size = grid_size
        self.transformer_dim = transformer_dim
        
        # 1. U-Net编码器，提取特征
        self.unet_encoder = nn.Sequential(
            # 第一阶段
            nn.Conv3d(in_channels, unet_features[0], kernel_size=3, padding=1),
            nn.BatchNorm3d(unet_features[0]),
            nn.ReLU(inplace=True),
            nn.Conv3d(unet_features[0], unet_features[0], kernel_size=3, padding=1),
            nn.BatchNorm3d(unet_features[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
            # 第二阶段
            nn.Conv3d(unet_features[0], unet_features[1], kernel_size=3, padding=1),
            nn.BatchNorm3d(unet_features[1]),
            nn.ReLU(inplace=True),
            nn.Conv3d(unet_features[1], unet_features[1], kernel_size=3, padding=1),
            nn.BatchNorm3d(unet_features[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
            # 第三阶段
            nn.Conv3d(unet_features[1], unet_features[2], kernel_size=3, padding=1),
            nn.BatchNorm3d(unet_features[2]),
            nn.ReLU(inplace=True),
            nn.Conv3d(unet_features[2], unet_features[2], kernel_size=3, padding=1),
            nn.BatchNorm3d(unet_features[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2)
        )
        
        # 特征映射到Transformer维度
        self.feature_proj = nn.Conv3d(unet_features[2], transformer_dim, kernel_size=1)
        
        # 3D位置编码
        self.pos_encoder = PositionalEncoding3D(transformer_dim, grid_size=(2, 2, 2))  # 经过三次池化后，尺寸变为16/8=2
        
        # Transformer编码器层
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dim_feedforward=transformer_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_transformer_layers)
        
        # U-Net解码器
        self.unet_decoder = nn.Sequential(
            # 上采样
            nn.ConvTranspose3d(transformer_dim, unet_features[2], kernel_size=2, stride=2),
            
            # 第三阶段解码
            nn.Conv3d(unet_features[2], unet_features[2], kernel_size=3, padding=1),
            nn.BatchNorm3d(unet_features[2]),
            nn.ReLU(inplace=True),
            nn.Conv3d(unet_features[2], unet_features[2], kernel_size=3, padding=1),
            nn.BatchNorm3d(unet_features[2]),
            nn.ReLU(inplace=True),
            
            # 上采样
            nn.ConvTranspose3d(unet_features[2], unet_features[1], kernel_size=2, stride=2),
            
            # 第二阶段解码
            nn.Conv3d(unet_features[1], unet_features[1], kernel_size=3, padding=1),
            nn.BatchNorm3d(unet_features[1]),
            nn.ReLU(inplace=True),
            nn.Conv3d(unet_features[1], unet_features[1], kernel_size=3, padding=1),
            nn.BatchNorm3d(unet_features[1]),
            nn.ReLU(inplace=True),
            
            # 上采样
            nn.ConvTranspose3d(unet_features[1], unet_features[0], kernel_size=2, stride=2),
            
            # 第一阶段解码
            nn.Conv3d(unet_features[0], unet_features[0], kernel_size=3, padding=1),
            nn.BatchNorm3d(unet_features[0]),
            nn.ReLU(inplace=True),
            nn.Conv3d(unet_features[0], unet_features[0], kernel_size=3, padding=1),
            nn.BatchNorm3d(unet_features[0]),
            nn.ReLU(inplace=True)
        )
        
        # 输出层
        self.final = nn.Conv3d(unet_features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        """前向传播"""
        # 1. U-Net编码器提取特征
        unet_features = self.unet_encoder(x)
        
        # 2. 特征映射到Transformer维度
        transformer_input = self.feature_proj(unet_features)
        
        # 3. 添加位置编码
        transformer_input = self.pos_encoder(transformer_input)
        
        # 4. Transformer处理
        batch_size = transformer_input.shape[0]
        depth, height, width = transformer_input.shape[2:5]
        
        # 转换为序列
        transformer_input = transformer_input.view(batch_size, self.transformer_dim, -1).permute(0, 2, 1)
        
        # Transformer编码
        transformer_output = self.transformer_encoder(transformer_input)
        
        # 转换回3D张量
        transformer_output = transformer_output.permute(0, 2, 1).view(batch_size, self.transformer_dim, depth, height, width)
        
        # 5. U-Net解码器恢复空间信息
        decoded = self.unet_decoder(transformer_output)
        
        # 6. 输出
        output = self.final(decoded)
        output = torch.sigmoid(output)
        
        return output

# 测试模型
if __name__ == "__main__":
    # 创建模型实例
    model = HybridUNetTransformer(in_channels=4, out_channels=1, grid_size=(16, 16, 16))
    
    # 打印模型结构
    print("混合模型结构:")
    print(model)
    
    # 测试输入输出形状
    input_tensor = torch.randn(1, 4, 16, 16, 16)
    output = model(input_tensor)
    
    print(f"\n输入形状: {input_tensor.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")