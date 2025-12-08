#!/usr/bin/env python3
"""
LeRobot数据集读取和检查脚本（简化版，无matplotlib依赖）

用于读取、检查转换后的LeRobot标准格式数据集
"""

import argparse
from pathlib import Path
import json
from typing import Optional
import sys

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import torch
    import numpy as np
except ImportError as e:
    print(f"❌ 缺少必要的库: {e}")
    print("请安装: pip install lerobot torch")
    sys.exit(1)


def load_dataset(dataset_path: str, repo_id: Optional[str] = None):
    """
    加载LeRobot数据集
    
    Args:
        dataset_path: 数据集根目录路径
        repo_id: 仓库ID（可选，默认使用路径名）
    """
    print(f"📂 加载数据集: {dataset_path}")
    
    dataset_path = Path(dataset_path).resolve()
    
    # 检查路径是否存在
    if not dataset_path.exists():
        print(f"❌ 路径不存在: {dataset_path}")
        sys.exit(1)
    
    # 检查是否是LeRobot数据集格式
    required_dirs = ['data', 'meta']
    if not all((dataset_path / d).exists() for d in required_dirs):
        print(f"❌ 不是有效的LeRobot数据集格式")
        print(f"   需要包含: {required_dirs}")
        sys.exit(1)
    
    # 检查info.json
    info_file = dataset_path / 'meta' / 'info.json'
    if not info_file.exists():
        print(f"⚠️  缺少 meta/info.json 文件")
        print(f"   数据集可能无法被LeRobot正确加载")
    
    try:
        # 如果没有指定repo_id，使用路径名
        if repo_id is None:
            repo_id = dataset_path.name
        
        print(f"🔧 加载参数:")
        print(f"   repo_id: {repo_id}")
        print(f"   root: {dataset_path}")
        
        # 直接加载本地数据集
        dataset = LeRobotDataset(
            repo_id=repo_id,
            root=str(dataset_path)
        )
        print(f"✓ 数据集加载成功")
        return dataset
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        print(f"\n调试信息:")
        print(f"  - 数据集路径: {dataset_path}")
        print(f"  - repo_id: {repo_id}")
        if info_file.exists():
            print(f"  - info.json 存在")
            with open(info_file, 'r') as f:
                try:
                    info = json.load(f)
                    print(f"  - features数: {len(info.get('features', {}))}")
                    print(f"  - data_path: {info.get('data_path', '未设置')}")
                except:
                    print(f"  - info.json 格式错误")
        print(f"\n目录结构:")
        for item in sorted(dataset_path.iterdir())[:10]:
            print(f"  - {item.name}")
        sys.exit(1)


def print_dataset_info(dataset):
    """
    打印数据集基本信息
    """
    print("\n" + "=" * 80)
    print("📊 数据集基本信息")
    print("=" * 80)
    
    print(f"总帧数: {len(dataset)}")
    
    # 获取元数据
    if hasattr(dataset, 'meta'):
        meta = dataset.meta
        print(f"\n📋 元数据:")
        if hasattr(meta, 'fps'):
            print(f"  - FPS: {meta.fps}")
        if hasattr(meta, 'robot_type'):
            print(f"  - 机器人类型: {meta.robot_type}")
        if hasattr(meta, 'total_episodes'):
            print(f"  - Episode数量: {meta.total_episodes}")
        if hasattr(meta, 'total_frames'):
            print(f"  - 总帧数: {meta.total_frames}")
    
    # 检查第一帧的数据结构
    print(f"\n🔍 数据结构（第一帧）:")
    first_frame = dataset[0]
    for key, value in first_frame.items():
        if isinstance(value, torch.Tensor):
            print(f"  - {key}: Tensor {value.shape} {value.dtype}")
        elif isinstance(value, np.ndarray):
            print(f"  - {key}: ndarray {value.shape} {value.dtype}")
        else:
            print(f"  - {key}: {type(value).__name__} = {value}")
    
    # 统计episode信息
    if 'episode_index' in first_frame:
        episode_indices = set()
        max_check = min(len(dataset), 10000)
        print(f"\n📦 统计Episode信息（检查前{max_check}帧）...")
        for i in range(max_check):
            episode_indices.add(int(dataset[i]['episode_index']))
        print(f"  - 检测到的Episode数: {len(episode_indices)}")
        print(f"  - Episode ID: {sorted(episode_indices)}")


def print_episode_info(dataset, episode_idx: int = 0):
    """
    打印指定episode的详细信息
    """
    print("\n" + "=" * 80)
    print(f"📦 Episode {episode_idx} 详细信息")
    print("=" * 80)
    
    # 找到该episode的所有帧
    episode_frames = []
    for i in range(len(dataset)):
        frame = dataset[i]
        if int(frame['episode_index']) == episode_idx:
            episode_frames.append(i)
        if len(episode_frames) > 0 and int(frame['episode_index']) > episode_idx:
            break  # 优化：已经过了该episode
    
    if not episode_frames:
        print(f"❌ 未找到Episode {episode_idx}")
        return
    
    print(f"总帧数: {len(episode_frames)}")
    print(f"帧索引范围: {episode_frames[0]} - {episode_frames[-1]}")
    
    # 检查第一帧
    first_frame = dataset[episode_frames[0]]
    
    # 打印任务信息
    if 'task' in first_frame:
        print(f"\n📝 任务信息:")
        print(f"  {first_frame['task']}")
    
    # 检查图像
    image_keys = [k for k in first_frame.keys() if 'image' in k.lower() or 'cam' in k.lower()]
    if image_keys:
        print(f"\n📷 图像信息:")
        for key in image_keys:
            img = first_frame[key]
            if isinstance(img, torch.Tensor):
                print(f"  - {key}: {img.shape} {img.dtype}")
    
    # 检查动作
    if 'action' in first_frame:
        action = first_frame['action']
        if isinstance(action, torch.Tensor):
            print(f"\n🎮 动作信息:")
            print(f"  - 形状: {action.shape}")
            print(f"  - 数据类型: {action.dtype}")
            print(f"  - 第一帧动作: {action.cpu().numpy()}")
            print(f"  - 最后帧动作: {dataset[episode_frames[-1]]['action'].cpu().numpy()}")
    
    # 检查状态
    if 'state' in first_frame:
        state = first_frame['state']
        if isinstance(state, torch.Tensor):
            print(f"\n🔧 状态信息:")
            print(f"  - 形状: {state.shape}")
            print(f"  - 数据类型: {state.dtype}")
            print(f"  - 第一帧状态: {state.cpu().numpy()}")


def save_frame_image(dataset, frame_idx: int, output_path: str, camera_key: Optional[str] = None):
    """
    保存指定帧的图像到文件
    
    Args:
        dataset: LeRobot数据集
        frame_idx: 帧索引
        output_path: 输出图像路径
        camera_key: 指定相机键（可选，默认使用第一个）
    """
    try:
        from PIL import Image
    except ImportError:
        print("❌ 需要安装PIL: pip install pillow")
        return
    
    frame = dataset[frame_idx]
    
    # 找到图像键
    image_keys = [k for k in frame.keys() if 'image' in k.lower() or 'cam' in k.lower()]
    
    if not image_keys:
        print("❌ 未找到图像数据")
        return
    
    # 选择相机
    if camera_key is None:
        camera_key = image_keys[0]
    elif camera_key not in image_keys:
        print(f"❌ 未找到相机 {camera_key}，可用: {image_keys}")
        return
    
    img = frame[camera_key]
    
    # 转换为numpy数组
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    
    # 调整维度顺序 (C, H, W) -> (H, W, C)
    if img.shape[0] in [1, 3, 4]:
        img = np.transpose(img, (1, 2, 0))
    
    # 归一化到0-255
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    
    # 保存
    Image.fromarray(img).save(output_path)
    print(f"✓ 图像已保存: {output_path}")


def print_frame_range_info(dataset_path: str):
    """
    打印frame_ranges_info.json的内容
    """
    info_file = Path(dataset_path) / 'frame_ranges_info.json'
    
    if not info_file.exists():
        print(f"⚠️  未找到 frame_ranges_info.json")
        return
    
    print("\n" + "=" * 80)
    print("📋 帧范围信息 (frame_ranges_info.json)")
    print("=" * 80)
    
    with open(info_file, 'r', encoding='utf-8') as f:
        info = json.load(f)
    
    print(f"\n总片段数: {info.get('total_ranges', 0)}")
    print(f"Pick动作: {info.get('pick_count', 0)}")
    print(f"Place动作: {info.get('place_count', 0)}")
    
    print(f"\n详细信息:")
    for r in info.get('frame_ranges', []):
        print(f"\n  [{r['id']}] {r['action_type'].upper()}")
        print(f"    关键帧: {r['keyframe_index']}")
        print(f"    帧范围: {r['frame_start']} - {r['frame_end']} ({r['num_frames']}帧)")
        print(f"    Episode: {r['episode_index']}, Frame: {r['frame_index']}")
        print(f"    原任务: {r['original_task']}")
        print(f"    新任务: {r['new_task']}")


def main():
    parser = argparse.ArgumentParser(
        description='LeRobot数据集读取和检查工具（简化版）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本信息查看
  python read_lerobot_dataset_simple.py --dataset-path ./cut_dataset
  
  # 查看特定episode
  python read_lerobot_dataset_simple.py --dataset-path ./cut_dataset --episode 0
  
  # 保存指定帧的图像
  python read_lerobot_dataset_simple.py --dataset-path ./cut_dataset --save-frame 0 --output frame_0.png
  
  # 查看帧范围信息
  python read_lerobot_dataset_simple.py --dataset-path ./cut_dataset --show-ranges
        """
    )
    
    parser.add_argument('--dataset-path', type=str, required=True,
                       help='LeRobot数据集路径')
    parser.add_argument('--repo-id', type=str, default=None,
                       help='仓库ID（可选）')
    parser.add_argument('--episode', type=int, default=None,
                       help='查看指定episode的详细信息')
    parser.add_argument('--save-frame', type=int, default=None,
                       help='保存指定帧的图像')
    parser.add_argument('--output', type=str, default='frame.png',
                       help='图像输出路径（配合--save-frame使用）')
    parser.add_argument('--camera', type=str, default=None,
                       help='指定相机键（可选）')
    parser.add_argument('--show-ranges', action='store_true',
                       help='显示frame_ranges_info.json内容')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("📖 LeRobot 数据集读取工具")
    print("=" * 80)
    
    # 显示帧范围信息
    if args.show_ranges:
        print_frame_range_info(args.dataset_path)
        print("\n" + "=" * 80)
        return
    
    # 加载数据集
    dataset = load_dataset(args.dataset_path, args.repo_id)
    
    # 打印基本信息
    print_dataset_info(dataset)
    
    # 查看特定episode
    if args.episode is not None:
        print_episode_info(dataset, args.episode)
    
    # 保存帧图像
    if args.save_frame is not None:
        print(f"\n💾 保存帧 {args.save_frame}...")
        save_frame_image(dataset, args.save_frame, args.output, args.camera)
    
    print("\n" + "=" * 80)
    print("✅ 完成")
    print("=" * 80)


if __name__ == '__main__':
    main()
