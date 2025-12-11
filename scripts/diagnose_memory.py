#!/usr/bin/env python3
"""
内存诊断脚本 - 估算所需的 batch_size
"""

import psutil
import sys


def get_available_memory_gb():
    """获取可用内存（GB）"""
    mem = psutil.virtual_memory()
    return mem.available / (1024 ** 3)


def estimate_batch_size(total_frames, avg_frames_per_episode=60):
    """
    估算合适的batch_size
    
    Args:
        total_frames: 数据集总帧数
        avg_frames_per_episode: 平均每个episode的帧数
    
    Returns:
        推荐的batch_size
    """
    available_memory = get_available_memory_gb()
    
    # 每帧大约占用1.5MB（包含2个摄像头图像）
    memory_per_frame_mb = 1.5
    
    # 保留40%的内存用于系统和其他开销
    usable_memory_gb = available_memory * 0.6
    
    # 计算可以同时处理的帧数
    max_frames_in_memory = (usable_memory_gb * 1024) / memory_per_frame_mb
    
    # 计算batch_size
    batch_size = int(max_frames_in_memory / avg_frames_per_episode)
    
    return max(10, min(batch_size, 300))  # 限制在10-300之间


def print_recommendations(total_frames):
    """打印推荐配置"""
    available_memory = get_available_memory_gb()
    total_memory = psutil.virtual_memory().total / (1024 ** 3)
    used_memory = psutil.virtual_memory().used / (1024 ** 3)
    
    print("=" * 60)
    print("🔍 系统内存分析")
    print("=" * 60)
    print(f"总内存: {total_memory:.2f} GB")
    print(f"已使用: {used_memory:.2f} GB ({(used_memory/total_memory)*100:.1f}%)")
    print(f"可用内存: {available_memory:.2f} GB")
    print()
    
    # 估算不同场景下的batch_size
    scenarios = [
        ("小型episode (30帧)", 30),
        ("中型episode (60帧)", 60),
        ("大型episode (100帧)", 100),
    ]
    
    print("📊 推荐配置:")
    print("-" * 60)
    
    for scenario_name, avg_frames in scenarios:
        batch_size = estimate_batch_size(total_frames, avg_frames)
        memory_usage = (batch_size * avg_frames * 1.5) / 1024  # GB
        
        print(f"\n{scenario_name}:")
        print(f"  推荐 batch_size: {batch_size}")
        print(f"  预计内存占用: ~{memory_usage:.2f} GB")
        print(f"  命令示例:")
        print(f"    python auto_cut_dataset.py --batch-size {batch_size} ...")
    
    print()
    print("=" * 60)
    print("⚠️  注意事项:")
    print("=" * 60)
    print("1. 如果内存不足，减小 batch_size")
    print("2. 如果内存充足，可以适当增加 batch_size 提升速度")
    print("3. 建议先用小的 batch_size 测试，确保稳定后再增加")
    print("4. 使用 htop 或 nvidia-smi 监控实际内存使用")
    print()
    
    # 内存不足警告
    if available_memory < 4:
        print("⚠️  警告: 可用内存不足4GB，建议:")
        print("   - 关闭其他占用内存的程序")
        print("   - 使用 --batch-size 20 或更小")
        print("   - 考虑分段处理数据集")
        print()


def main():
    if len(sys.argv) > 1:
        try:
            total_frames = int(sys.argv[1])
        except ValueError:
            print("用法: python diagnose_memory.py [总帧数]")
            print("示例: python diagnose_memory.py 273465")
            sys.exit(1)
    else:
        total_frames = 273465  # 默认值
        print(f"未指定总帧数，使用默认值: {total_frames}")
        print()
    
    print_recommendations(total_frames)
    
    # 交互式建议
    print("🤔 需要更具体的建议？")
    print("   运行: python diagnose_memory.py <你的数据集总帧数>")
    print()


if __name__ == '__main__':
    main()
