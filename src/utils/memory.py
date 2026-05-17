import torch
import time

# 指定使用 GPU 1
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

if not torch.cuda.is_available():
    print("CUDA 不可用，程序退出")
    exit(1)

print(f"使用设备: {device}")
print(f"GPU 名称: {torch.cuda.get_device_name(1)}")
print(f"可用显存: {torch.cuda.get_device_properties(1).total_memory / 1024**3:.2f} GB")

# 计算需要分配的张量大小
# 目标: 30 GB
# float32 每个数占 4 bytes
# 所需元素数 = 30 * 1024^3 / 4 ≈ 8,053,063,680 个元素

target_gb = 30
bytes_per_element = 4  # float32
num_elements = int(target_gb * 1024**3 / bytes_per_element)

print(f"目标分配显存: {target_gb} GB")
print(f"需要张量形状: 约 {num_elements:,} 个元素")

# 创建一个接近目标大小的张量
# 使用一个三维张量便于调整大小
size = int(round(num_elements ** (1/3)))
tensor = torch.randn(size, size, size, dtype=torch.float32, device=device)

actual_gb = tensor.numel() * bytes_per_element / 1024**3
print(f"实际分配显存: {actual_gb:.2f} GB")
print(f"张量形状: {tensor.shape}")
print(f"张量元素数: {tensor.numel():,}")

# 保持程序运行，让显存持续占用
print("\n正在占用显存... 按 Ctrl+C 退出")
try:
    while True:
        # 定期检查显存使用情况
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        print(f"已分配: {allocated:.2f} GB | 已保留: {reserved:.2f} GB", end="\r")
        time.sleep(2)
except KeyboardInterrupt:
    print("\n释放显存，退出程序")
    del tensor
    torch.cuda.empty_cache()