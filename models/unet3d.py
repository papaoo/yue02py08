import torch
import torch.nn as nn

class UNet3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super().__init__()
        
        # 编码器部分
        self.enc1 = self.conv_block(in_channels, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        
        # 瓶颈层
        self.bottleneck = self.conv_block(128, 256)
        
        # 解码器部分
        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(256, 128)
        
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(128, 64)
        
        self.up1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(64, 32)
        
        # 输出层
        self.final = nn.Conv3d(32, out_channels, kernel_size=1)
        
        # 池化层
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
    
    def conv_block(self, in_ch, out_ch):
        """3D卷积块，包含两次3D卷积+批标准化+ReLU激活"""
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """前向传播"""
        # 编码器部分
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        
        # 瓶颈层
        bottleneck = self.bottleneck(self.pool(enc3))
        
        # 解码器部分
        dec3 = self.up3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)  # 跳跃连接
        dec3 = self.dec3(dec3)
        
        dec2 = self.up2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)  # 跳跃连接
        dec2 = self.dec2(dec2)
        
        dec1 = self.up1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)  # 跳跃连接
        dec1 = self.dec1(dec1)
        
        # 输出层
        output = self.final(dec1)
        output = torch.sigmoid(output)  # 使用sigmoid激活，输出概率值
        
        return output

# 测试模型
if __name__ == "__main__":
    # 创建模型实例
    model = UNet3D(in_channels=4, out_channels=1)
    
    # 打印模型结构
    print("3D U-Net模型结构:")
    print(model)
    
    # 测试输入输出形状
    input_tensor = torch.randn(1, 4, 16, 16, 16)  # [batch_size, channels, depth, height, width]
    output = model(input_tensor)
    
    print(f"\n输入形状: {input_tensor.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")