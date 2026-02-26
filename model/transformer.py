import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding3D(nn.Module):
    """3D位置编码"""
    def __init__(self, d_model, grid_size):
        super().__init__()
        self.d_model = d_model
        depth, height, width = grid_size
        
        # 创建位置编码表 (1, d_model, depth, height, width)，添加batch维度
        pe = torch.zeros(1, d_model, depth, height, width)
        
        # 创建位置索引网格
        d_pos = torch.arange(depth)
        h_pos = torch.arange(height)
        w_pos = torch.arange(width)
        d_grid, h_grid, w_grid = torch.meshgrid(d_pos, h_pos, w_pos, indexing='ij')
        
        # 位置编码公式
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        # 分别对三个维度应用正弦和余弦编码
        for i in range(0, d_model, 2):
            if i + 1 < d_model:
                pe[0, i, :, :, :] = torch.sin(d_grid * div_term[i//2])
                pe[0, i+1, :, :, :] = torch.cos(d_grid * div_term[i//2])
                
                pe[0, i, :, :, :] += torch.sin(h_grid * div_term[i//2])
                pe[0, i+1, :, :, :] += torch.cos(h_grid * div_term[i//2])
                
                pe[0, i, :, :, :] += torch.sin(w_grid * div_term[i//2])
                pe[0, i+1, :, :, :] += torch.cos(w_grid * div_term[i//2])
            else:
                # 处理奇数d_model
                pe[0, i, :, :, :] = torch.sin(d_grid * div_term[i//2])
                pe[0, i, :, :, :] += torch.sin(h_grid * div_term[i//2])
                pe[0, i, :, :, :] += torch.sin(w_grid * div_term[i//2])
        
        # 注册为缓冲区，不参与训练
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """添加位置编码"""
        return x + self.pe

class Transformer3D(nn.Module):
    """3D Transformer模型，用于路径规划"""
    def __init__(self, in_channels=4, out_channels=1, grid_size=(16, 16, 16), 
                 embed_dim=256, num_heads=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.grid_size = grid_size
        self.embed_dim = embed_dim
        
        # 输入嵌入层：将输入通道转换为嵌入维度
        self.input_embed = nn.Conv3d(in_channels, embed_dim, kernel_size=1)
        
        # 3D位置编码
        self.pos_encoder = PositionalEncoding3D(embed_dim, grid_size)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True  # 使用batch_first=True，输入形状为(batch, seq_len, embed_dim)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层：将嵌入维度转换为输出通道
        self.output_proj = nn.Conv3d(embed_dim, out_channels, kernel_size=1)
    
    def forward(self, x):
        """前向传播"""
        batch_size = x.shape[0]
        depth, height, width = self.grid_size
        
        # 输入嵌入
        x = self.input_embed(x)  # 形状: (batch_size, embed_dim, depth, height, width)
        
        # 添加位置编码
        x = self.pos_encoder(x)  # 形状: (batch_size, embed_dim, depth, height, width)
        
        # 将3D张量转换为序列: (batch_size, embed_dim, depth*height*width) -> (batch_size, depth*height*width, embed_dim)
        x = x.view(batch_size, self.embed_dim, -1).permute(0, 2, 1)
        
        # Transformer编码器
        x = self.transformer_encoder(x)  # 形状: (batch_size, depth*height*width, embed_dim)
        
        # 将序列转换回3D张量: (batch_size, depth*height*width, embed_dim) -> (batch_size, embed_dim, depth, height, width)
        x = x.permute(0, 2, 1).view(batch_size, self.embed_dim, depth, height, width)
        
        # 输出投影
        x = self.output_proj(x)  # 形状: (batch_size, out_channels, depth, height, width)
        x = torch.sigmoid(x)  # 使用sigmoid激活，输出概率值
        
        return x

# 测试模型
if __name__ == "__main__":
    # 创建模型实例
    model = Transformer3D(in_channels=4, out_channels=1, grid_size=(16, 16, 16))
    
    # 打印模型结构
    print("3D Transformer模型结构:")
    print(model)
    
    # 测试输入输出形状
    input_tensor = torch.randn(1, 4, 16, 16, 16)  # [batch_size, channels, depth, height, width]
    output = model(input_tensor)
    
    print(f"\n输入形状: {input_tensor.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")