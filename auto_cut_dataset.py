#!/usr/bin/env python3
"""
自动化的Pick/Place数据集裁剪和转换脚本

流程：
1. 加载LeRobot数据集
2. 检测夹爪状态变化关键帧
3. 提取前后各30帧
4. 使用LLM生成任务描述
5. 转换为LeRobot格式保存
"""

import torch
import numpy as np
import argparse
from pathlib import Path
import json
import sys
from typing import Optional
import time
from datetime import datetime

# 添加模块路径
sys.path.insert(0, str(Path(__file__).parent))

from gripper_detector import analyze_gripper_changes
from task_description_generator import TaskDescriptionGenerator
from dataset_cutter import cut_and_convert_dataset


def load_lerobot_dataset(dataset_path: str = None):
    """
    加载LeRobot数据集
    """
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        print("❌ 需要安装LeRobot库: pip install lerobot")
        sys.exit(1)
    
    if dataset_path is None:
        dataset_path = '/home/dongyingyibadao/HuggingFaceVLA_cus/libero'
    
    print(f"📂 加载数据集: {dataset_path}")
    
    try:
        # 使用与data_dealer相同的加载方式
        dataset = LeRobotDataset(
            repo_id="HuggingFaceVLA_cus/libero",
            root=str(dataset_path)
        )
        print(f"✓ 数据集加载成功，共 {len(dataset)} 帧")
        return dataset
    except Exception as e:
        print(f"❌ 加载数据集失败: {e}")
        sys.exit(1)


def analyze_and_extract(dataset,
                       start_idx: int = 0,
                       end_idx: int = 10000,
                       before_frames: int = 30,
                       after_frames: int = 30) -> tuple:
    """
    分析数据集并提取关键帧
    """
    print(f"\n🔍 分析数据集 ({start_idx} - {end_idx})...")
    print(f"  - 关键帧前: {before_frames} 帧")
    print(f"  - 关键帧后: {after_frames} 帧")
    
    changes, frame_ranges = analyze_gripper_changes(
        dataset, 
        start_idx, 
        end_idx, 
        before_frames=before_frames,
        after_frames=after_frames,
        merge=False
    )
    
    return changes, frame_ranges


def generate_task_descriptions(frame_ranges: list,
                               dataset = None,
                               provider: str = 'local',
                               api_key: Optional[str] = None,
                               api_base: Optional[str] = None,
                               api_version: Optional[str] = None,
                               model: Optional[str] = None,
                               fast_mode: bool = False,
                               checkpoint_dir: Optional[Path] = None,
                               resume_from: Optional[str] = None) -> list:
    """
    为关键帧生成任务描述（支持断点续传）
    
    Args:
        checkpoint_dir: 检查点保存目录
        resume_from: 从检查点文件恢复
    """
    mode_str = "快速模式(2帧)" if fast_mode else "精细模式(6帧)"
    print(f"\n🤖 生成任务描述... [{mode_str}]")
    
    # 准备检查点目录
    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"💾 检查点保存: {checkpoint_dir}")
    
    # 尝试从检查点恢复
    start_idx = 0
    completed_ranges = []
    
    if resume_from and Path(resume_from).exists():
        print(f"\n📖 从检查点恢复: {resume_from}")
        try:
            with open(resume_from, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            completed_ranges = checkpoint_data.get('completed_ranges', [])
            start_idx = checkpoint_data.get('last_index', 0) + 1
            
            print(f"✓ 已恢复 {len(completed_ranges)} 个已完成的任务描述")
            print(f"✓ 从索引 {start_idx}/{len(frame_ranges)} 继续处理")
        except Exception as e:
            print(f"⚠️  读取检查点失败: {e}，从头开始")
            start_idx = 0
            completed_ranges = []
    
    kwargs = {'provider': provider}
    if api_key:
        kwargs['api_key'] = api_key
    if api_base:
        kwargs['api_base'] = api_base
    if api_version:
        kwargs['api_version'] = api_version
    if model:
        kwargs['model'] = model
    if provider.lower() == 'gpt':
        kwargs['fast_mode'] = fast_mode
    
    generator = TaskDescriptionGenerator(**kwargs)
    
    # 带断点保存的描述生成
    ranges_with_desc = generator.generate_descriptions(
        frame_ranges, 
        dataset=dataset,
        start_index=start_idx,
        completed_ranges=completed_ranges,
        checkpoint_dir=checkpoint_dir
    )
    
    return ranges_with_desc


def save_frame_ranges_info(frame_ranges: list, output_path: Path):
    """
    保存帧范围信息为JSON
    """
    def convert_to_serializable(val):
        """将任何值转换为JSON可序列化的格式"""
        if isinstance(val, torch.Tensor):
            return int(val.item()) if val.numel() == 1 else val.tolist()
        elif isinstance(val, np.ndarray):
            return val.tolist()
        elif isinstance(val, (int, float, str, bool, type(None))):
            return val
        else:
            return str(val)
    
    info = {
        'total_ranges': len(frame_ranges),
        'frame_ranges': []
    }
    
    pick_count = 0
    place_count = 0
    
    for r in frame_ranges:
        if r['action_type'] == 'pick':
            pick_count += 1
        else:
            place_count += 1
        
        info['frame_ranges'].append({
            'id': len(info['frame_ranges']),
            'keyframe_index': convert_to_serializable(r['keyframe_index']),
            'action_type': r['action_type'],
            'frame_start': convert_to_serializable(r['frame_start']),
            'frame_end': convert_to_serializable(r['frame_end']),
            'num_frames': convert_to_serializable(r['num_frames']),
            'original_task': str(r['task']),
            'new_task': str(r.get('new_task', r['task'])),
            'episode_index': convert_to_serializable(r['episode_index']),
            'frame_index': convert_to_serializable(r['frame_index'])
        })
    
    info['pick_count'] = pick_count
    info['place_count'] = place_count
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 保存帧范围信息: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='自动化Pick/Place数据集裁剪和转换'
    )
    parser.add_argument('--dataset-path', type=str, default=None,
                       help='LeRobot数据集路径')
    parser.add_argument('--output-dir', type=str, 
                       default='./cut_dataset',
                       help='输出目录')
    parser.add_argument('--start-idx', type=int, default=0,
                       help='开始索引')
    parser.add_argument('--end-idx', type=int, default=None,
                       help='结束索引（默认：处理所有数据）')
    parser.add_argument('--max-episodes', type=int, default=None,
                       help='最多保存的episode数量')
    parser.add_argument('--before-frames', type=int, default=30,
                       help='关键帧前取的帧数')
    parser.add_argument('--after-frames', type=int, default=30,
                       help='关键帧后取的帧数')
    parser.add_argument('--save-mode', type=str, default='lerobot',
                       choices=['image', 'lerobot', 'both'],
                       help='保存模式: image(图片), lerobot(Parquet), both(两者)')
    parser.add_argument('--llm-provider', type=str, default='local',
                       choices=['local', 'qwen', 'deepseek', 'gpt'],
                       help='LLM提供者')
    parser.add_argument('--llm-api-key', type=str, default=None,
                       help='LLM API密钥')
    parser.add_argument('--llm-api-base', type=str, default=None,
                       help='LLM API基础URL (用于自定义/代理服务)')
    parser.add_argument('--llm-api-version', type=str, default=None,
                       help='LLM API版本 (用于Azure OpenAI)')
    parser.add_argument('--llm-model', type=str, default=None,
                       help='指定LLM模型名称 (例如: gpt-4o, gpt-4-turbo, o1-preview)')
    parser.add_argument('--llm-fast-mode', action='store_true',
                       help='GPT快速模式：仅上传2帧图像(cam1首尾帧)，处理速度更快')
    parser.add_argument('--checkpoint-interval', type=int, default=10,
                       help='检查点保存间隔（每处理多少个保存一次，默认10）')
    parser.add_argument('--resume-from', type=str, default=None,
                       help='从检查点文件恢复（例如：./cut_dataset/checkpoints/checkpoint_latest.json）')
    parser.add_argument('--skip-cutting', action='store_true',
                       help='跳过数据集裁剪，仅生成分析')
    parser.add_argument('--load-ranges', type=str, default=None,
                       help='加载之前保存的帧范围信息')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 Pick/Place 自动化数据集裁剪和转换")
    print("=" * 80)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载或生成帧范围信息
    ranges_info_file = output_dir / 'frame_ranges_info.json'
    
    if args.load_ranges:
        print(f"\n📖 加载之前保存的帧范围信息: {args.load_ranges}")
        with open(args.load_ranges, 'r') as f:
            ranges_info = json.load(f)
        # 重构frame_ranges
        frame_ranges = ranges_info['frame_ranges']
    else:
        # 加载数据集
        dataset = load_lerobot_dataset(args.dataset_path)
        
        # 如果没有指定 end_idx，使用数据集总长度
        end_idx = args.end_idx if args.end_idx is not None else len(dataset)
        
        print(f"📊 处理范围: {args.start_idx} - {end_idx} (共 {end_idx - args.start_idx} 帧)")
        if args.end_idx is None:
            print(f"   ℹ️  未指定 --end-idx，将处理所有数据")
        
        # 分析和提取
        changes, frame_ranges = analyze_and_extract(
            dataset, 
            args.start_idx, 
            end_idx,
            before_frames=args.before_frames,
            after_frames=args.after_frames
        )
        
        # 生成任务描述
        checkpoint_dir = output_dir / 'checkpoints' if output_dir else None
        
        frame_ranges = generate_task_descriptions(
            frame_ranges,
            dataset=dataset,
            provider=args.llm_provider,
            api_key=args.llm_api_key,
            api_base=args.llm_api_base,
            api_version=args.llm_api_version,
            model=args.llm_model,
            fast_mode=args.llm_fast_mode,
            checkpoint_dir=checkpoint_dir,
            resume_from=args.resume_from
        )
        
        # 保存帧范围信息
        save_frame_ranges_info(frame_ranges, ranges_info_file)
    
    # 裁剪数据集
    if not args.skip_cutting:
        print(f"\n💾 开始裁剪和转换数据集...")
        print(f"📦 保存模式: {args.save_mode}")
        
        # 重新加载数据集（如果没有加载的话）
        if args.load_ranges:
            dataset = load_lerobot_dataset(args.dataset_path)
        
        output_path = cut_and_convert_dataset(
            dataset,
            frame_ranges,
            str(output_dir),
            save_mode=args.save_mode,
            max_episodes=args.max_episodes
        )
        
        print(f"\n✅ 数据集裁剪和转换完成!")
        print(f"📂 输出目录: {output_path}")
        
        if args.save_mode == 'image':
            print(f"📋 图片模式: 可以直接查看 {output_path}/images/ 目录下的图片")
        elif args.save_mode == 'lerobot':
            print(f"📋 LeRobot模式: 可以使用LeRobotDataset加载训练")
        else:
            print(f"📋 两种模式都已保存")
    else:
        print(f"\n⏭️  已跳过数据集裁剪步骤")
        print(f"📋 帧范围信息已保存: {ranges_info_file}")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
