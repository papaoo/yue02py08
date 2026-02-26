# 测试混合模型是否能正常导入和创建
print("开始测试混合模型...")

# 测试导入
try:
    from models.hybrid import HybridUNetTransformer
    print("✓ 混合模型导入成功")
    
    # 测试模型创建
    import torch
    model = HybridUNetTransformer(in_channels=4, out_channels=1, grid_size=(16, 16, 16))
    print("✓ 混合模型创建成功")
    
    # 测试输入输出
    input_tensor = torch.randn(1, 4, 16, 16, 16)
    output = model(input_tensor)
    print(f"✓ 模型前向传播成功")
    print(f"  输入形状: {input_tensor.shape}")
    print(f"  输出形状: {output.shape}")
    print(f"  输出范围: [{output.min():.4f}, {output.max():.4f}]")
    
    print("\n混合模型测试完成！")
    
except Exception as e:
    print(f"✗ 测试失败: {e}")