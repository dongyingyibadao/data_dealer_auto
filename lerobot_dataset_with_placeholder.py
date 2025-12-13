#!/usr/bin/env python3
"""
带占位符的 LeRobot Dataset 包装器

功能：
1. 在同一 episode 的不同 segment 之间自动插入占位符帧
2. 占位符帧标记为 is_placeholder=True，包含特殊标识
3. 不同 episode 之间不插入占位符
4. 完全兼容原始 LeRobotDataset 的所有接口

使用场景：
适用于 motion_planning 系统，需要明确标识同一 episode 内的跳跃动作边界
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
from collections import defaultdict
import json
import copy

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    print("❌ 请先安装 lerobot: pip install lerobot")
    raise


class AdjustedEpisodesWrapper:
    """
    Episodes包装器，动态调整dataset_from_index和dataset_to_index
    """
    def __init__(self, original_episodes, adjusted_ranges):
        self._original_episodes = original_episodes
        self._adjusted_ranges = adjusted_ranges
    
    def __len__(self):
        return len(self._original_episodes)
    
    def __getitem__(self, idx):
        """返回调整后的episode元数据"""
        original_ep = self._original_episodes[idx]
        
        if idx in self._adjusted_ranges:
            # 创建一个新字典，包含调整后的索引
            adjusted_ep = dict(original_ep)
            adjusted_ep['dataset_from_index'] = self._adjusted_ranges[idx]['dataset_from_index']
            adjusted_ep['dataset_to_index'] = self._adjusted_ranges[idx]['dataset_to_index']
            return adjusted_ep
        
        return original_ep
    
    def __iter__(self):
        """支持迭代"""
        for idx in range(len(self)):
            yield self[idx]


class AdjustedMetadataWrapper:
    """
    Metadata包装器，返回调整后的episodes
    """
    def __init__(self, original_meta, adjusted_ranges):
        self._original_meta = original_meta
        self._adjusted_episodes = AdjustedEpisodesWrapper(original_meta.episodes, adjusted_ranges)
    
    @property
    def episodes(self):
        """返回调整后的episodes"""
        return self._adjusted_episodes
    
    def __getattr__(self, name):
        """其他属性直接从原始meta获取"""
        return getattr(self._original_meta, name)


class LeRobotDatasetWithPlaceholder:
    """
    LeRobot Dataset 的包装器，在同一 episode 的不同 segment 之间插入占位符
    
    占位符特性：
    - is_placeholder=True 标记
    - action 全为 -999 (特殊标识值)
    - observation 使用前一帧的数据
    - episode_index 和 task 保持不变
    """
    
    def __init__(
        self,
        repo_id: str,
        root: str,
        placeholder_action_value: float = -999.0,
        **kwargs
    ):
        """
        初始化带占位符的数据集
        
        Args:
            repo_id: 数据集ID（通常是数据集名称）
            root: 数据集根目录
            placeholder_action_value: 占位符的action值（默认-999）
            **kwargs: 传递给原始 LeRobotDataset 的其他参数
        """
        print(f"🔧 初始化带占位符的 LeRobot Dataset...")
        print(f"   repo_id: {repo_id}")
        print(f"   root: {root}")
        
        # 加载原始数据集
        self.original_dataset = LeRobotDataset(repo_id=repo_id, root=root, **kwargs)
        self.placeholder_value = placeholder_action_value
        self.root = Path(root)
        
        # 分析 episode 和 segment 结构
        self._analyze_episode_structure()
        
        # 构建索引映射（原始索引 -> 新索引，插入占位符后）
        self._build_index_mapping()
        
        # 构建调整后的meta信息（方案1：动态更新meta）
        self._build_adjusted_meta()
        
        print(f"✅ 数据集加载完成")
        print(f"   原始帧数: {len(self.original_dataset)}")
        print(f"   新增占位符: {self.num_placeholders}")
        print(f"   总帧数: {len(self)}")
        print(f"   Episode数: {self.num_episodes}")
        
    def _analyze_episode_structure(self):
        """分析每个 episode 包含哪些 segment（数据文件）"""
        print("🔍 分析 episode 结构...")
        
        # 读取 episode 元数据
        episodes_meta = self.original_dataset.meta.episodes
        
        # 按 chunk_index 分组（同一个原始 episode 的不同 segment）
        # chunk_index 表示原始的 episode，多个 episode_index 可能属于同一个 chunk_index
        self.episode_segments = defaultdict(list)
        
        for ep_idx in range(len(episodes_meta)):
            ep_meta = episodes_meta[ep_idx]
            episode_index = ep_meta['episode_index']
            chunk_index = ep_meta['data/chunk_index']
            file_index = ep_meta['data/file_index']
            from_idx = ep_meta['dataset_from_index']
            to_idx = ep_meta['dataset_to_index']
            
            # 使用 chunk_index 作为分组键（表示同一个原始 episode）
            self.episode_segments[chunk_index].append({
                'episode_index': episode_index,
                'file_index': file_index,
                'chunk_index': chunk_index,
                'from_idx': from_idx,
                'to_idx': to_idx,
                'length': to_idx - from_idx + 1
            })
        
        # 对每个原始 episode 的 segment 按 from_idx 排序
        for chunk_idx in self.episode_segments:
            self.episode_segments[chunk_idx].sort(key=lambda x: x['from_idx'])
        
        # 统计需要插入的占位符数量
        self.placeholder_positions = []  # 存储所有占位符的插入位置
        
        for chunk_idx, segments in self.episode_segments.items():
            if len(segments) > 1:
                # 有多个 segment，需要在相邻 segment 之间插入占位符
                for i in range(len(segments) - 1):
                    insert_after_idx = segments[i]['to_idx']  # 在第 i 个 segment 的最后一帧之后插入
                    next_segment_first_idx = segments[i+1]['from_idx']
                    
                    self.placeholder_positions.append({
                        'chunk_index': chunk_idx,  # 原始 episode
                        'episode_index': segments[i]['episode_index'],  # 第 i 个 segment 的 episode_index
                        'next_episode_index': segments[i+1]['episode_index'],  # 第 i+1 个 segment 的 episode_index
                        'insert_after_original_idx': insert_after_idx,
                        'segment_boundary': (i, i+1)  # 在第i和第i+1个segment之间
                    })
        
        self.num_placeholders = len(self.placeholder_positions)
        
        print(f"   分析完成:")
        print(f"   - 原始 Episodes (chunk_index): {len(self.episode_segments)}")
        print(f"   - 切分后的 Segments: {sum(len(segs) for segs in self.episode_segments.values())}")
        print(f"   - 多 Segment Episodes: {sum(1 for segs in self.episode_segments.values() if len(segs) > 1)}")
        print(f"   - 需插入占位符: {self.num_placeholders} 个")
        
        # 打印详细的 segment 结构
        for chunk_idx, segments in sorted(self.episode_segments.items()):
            if len(segments) > 1:
                print(f"   原始 Episode {chunk_idx} (chunk_index): {len(segments)} segments")
                for i, seg in enumerate(segments):
                    print(f"      Segment {i} (episode_index={seg['episode_index']}): frames {seg['from_idx']}-{seg['to_idx']} (length={seg['length']})")
    
    def _build_index_mapping(self):
        """
        构建新旧索引的映射关系
        
        新索引 = 原始索引 + 之前插入的占位符数量
        """
        # 对占位符位置按原始索引排序
        self.placeholder_positions.sort(key=lambda x: x['insert_after_original_idx'])
        
        # 计算每个新索引对应的原始索引（或占位符标记）
        self.new_to_original_idx = []  # 新索引 -> (原始索引, is_placeholder)
        
        original_idx = 0
        placeholder_idx = 0
        
        while original_idx < len(self.original_dataset):
            # 添加当前原始帧
            self.new_to_original_idx.append((original_idx, False))
            
            # 检查是否需要在这个位置后插入占位符
            if placeholder_idx < len(self.placeholder_positions):
                placeholder_info = self.placeholder_positions[placeholder_idx]
                if original_idx == placeholder_info['insert_after_original_idx']:
                    # 插入占位符
                    self.new_to_original_idx.append((-1, True, placeholder_info))
                    placeholder_idx += 1
            
            original_idx += 1
        
        print(f"🗺️  索引映射构建完成: {len(self.new_to_original_idx)} 个新索引")
    
    def _build_adjusted_meta(self):
        """
        构建调整后的meta信息（方案1实现）
        
        根据placeholder的插入位置，重新计算所有episode的dataset_from_index和dataset_to_index
        使meta信息与实际数据索引保持一致
        """
        print("📝 构建调整后的meta信息...")
        
        # 构建原始索引到新索引的映射表
        # original_to_new[original_idx] = new_idx
        self.original_to_new_idx = {}
        for new_idx, mapping in enumerate(self.new_to_original_idx):
            if not mapping[1]:  # 不是placeholder
                original_idx = mapping[0]
                self.original_to_new_idx[original_idx] = new_idx
        
        # 为每个episode存储调整后的索引
        self._adjusted_episode_ranges = {}
        episodes_meta = self.original_dataset.meta.episodes
        
        for ep_idx in range(len(episodes_meta)):
            ep_meta = episodes_meta[ep_idx]
            original_from = ep_meta['dataset_from_index']
            original_to = ep_meta['dataset_to_index']
            
            # 使用映射表直接获取调整后的索引
            adjusted_from = self.original_to_new_idx.get(original_from, original_from)
            adjusted_to = self.original_to_new_idx.get(original_to, original_to)
            
            # 存储调整后的范围
            self._adjusted_episode_ranges[ep_idx] = {
                'dataset_from_index': adjusted_from,
                'dataset_to_index': adjusted_to,
                'offset': adjusted_from - original_from
            }
            
            # 调试信息（前5个episode）
            if ep_idx < 5:
                offset = adjusted_from - original_from
                print(f"   Episode {ep_idx}: {original_from:>3}-{original_to:<3} -> {adjusted_from:>3}-{adjusted_to:<3} (偏移+{offset})")
        
        print(f"✅ Meta信息更新完成，所有episode的索引已调整")
    
    def _create_placeholder_frame(self, previous_frame: Dict[str, Any], episode_index: int) -> Dict[str, Any]:
        """
        创建占位符帧
        
        Args:
            previous_frame: 前一帧的数据（用于复制观测）
            episode_index: 当前 episode 索引
            
        Returns:
            占位符帧数据
        """
        placeholder = {}
        
        # 复制观测数据（图像和状态）
        for key in previous_frame.keys():
            if key.startswith('observation.'):
                placeholder[key] = previous_frame[key].clone() if torch.is_tensor(previous_frame[key]) else previous_frame[key]
        
        # 设置特殊的 action 值
        action_shape = previous_frame['action'].shape
        placeholder['action'] = torch.full(action_shape, self.placeholder_value, dtype=previous_frame['action'].dtype)
        
        # 复制元数据，但标记为占位符
        if 'timestamp' in previous_frame:
            placeholder['timestamp'] = previous_frame['timestamp']
        if 'episode_index' in previous_frame:
            placeholder['episode_index'] = previous_frame['episode_index']
        if 'task_index' in previous_frame:
            placeholder['task_index'] = previous_frame['task_index']
        if 'task' in previous_frame:
            placeholder['task'] = previous_frame['task']
        
        # 添加占位符标记
        placeholder['is_placeholder'] = torch.tensor(True)
        placeholder['frame_index'] = torch.tensor(-1)  # 占位符没有有效的 frame_index
        
        return placeholder
    
    def __len__(self) -> int:
        """返回总帧数（包括占位符）"""
        return len(self.new_to_original_idx)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取指定索引的帧数据
        
        Args:
            idx: 新索引（包含占位符后的索引）
            
        Returns:
            帧数据字典
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")
        
        mapping = self.new_to_original_idx[idx]
        
        if mapping[1]:  # is_placeholder
            # 这是一个占位符
            placeholder_info = mapping[2]
            chunk_index = placeholder_info['chunk_index']
            episode_index = placeholder_info['episode_index']
            
            # 获取前一帧数据（用于复制观测）
            previous_original_idx = placeholder_info['insert_after_original_idx']
            previous_frame = self.original_dataset[previous_original_idx]
            
            # 创建占位符帧
            return self._create_placeholder_frame(previous_frame, episode_index)
        else:
            # 这是原始数据帧
            original_idx = mapping[0]
            frame = self.original_dataset[original_idx]
            
            # 添加标记：不是占位符
            frame['is_placeholder'] = torch.tensor(False)
            
            return frame
    
    @property
    def num_episodes(self) -> int:
        """返回原始 episode 总数（按 chunk_index 计算）"""
        return len(self.episode_segments)
    
    @property
    def num_segments(self) -> int:
        """返回切分后的 segment 总数（按 episode_index 计算）"""
        return self.original_dataset.num_episodes
    
    @property
    def meta(self):
        """返回调整后的元数据（包含placeholder偏移）"""
        if not hasattr(self, '_meta_wrapper'):
            self._meta_wrapper = AdjustedMetadataWrapper(
                self.original_dataset.meta,
                self._adjusted_episode_ranges
            )
        return self._meta_wrapper
    
    @property
    def original_meta(self):
        """返回原始数据集的元数据（未调整）"""
        return self.original_dataset.meta
    
    @property
    def hf_dataset(self):
        """返回原始数据集的 HuggingFace Dataset"""
        return self.original_dataset.hf_dataset
    
    def get_episode_info(self, chunk_idx: int) -> Dict:
        """
        获取指定原始 episode（chunk_index）的详细信息
        
        Args:
            chunk_idx: Chunk 索引（原始 episode）
            
        Returns:
            包含 segment 信息和占位符位置的字典
        """
        if chunk_idx not in self.episode_segments:
            raise ValueError(f"Chunk {chunk_idx} not found")
        
        segments = self.episode_segments[chunk_idx]
        
        # 找出这个 chunk 中的占位符位置
        placeholders = [p for p in self.placeholder_positions if p['chunk_index'] == chunk_idx]
        
        return {
            'chunk_index': chunk_idx,
            'num_segments': len(segments),
            'segments': segments,
            'num_placeholders': len(placeholders),
            'placeholder_positions': placeholders
        }
    
    def print_episode_structure(self, chunk_idx: Optional[int] = None):
        """
        打印 episode 结构信息
        
        Args:
            chunk_idx: 指定原始 episode（chunk_index），None 则打印所有
        """
        if chunk_idx is not None:
            chunks = [chunk_idx]
        else:
            chunks = sorted(self.episode_segments.keys())
        
        print("\n" + "="*80)
        print("📊 Episode 结构信息")
        print("="*80)
        
        for ch_idx in chunks:
            info = self.get_episode_info(ch_idx)
            print(f"\n原始 Episode {ch_idx} (chunk_index={ch_idx}):")
            print(f"  切分为 {info['num_segments']} 个 Segments:")
            
            for i, seg in enumerate(info['segments']):
                print(f"    Segment {i} (episode_index={seg['episode_index']}):")
                print(f"      原始帧范围: {seg['from_idx']}-{seg['to_idx']}")
                print(f"      帧数: {seg['length']}")
                print(f"      文件: data/episode_{seg['chunk_index']}/segment_{seg['file_index']}.parquet")
            
            if info['num_placeholders'] > 0:
                print(f"  占位符: {info['num_placeholders']} 个")
                for p in info['placeholder_positions']:
                    print(f"    在原始索引 {p['insert_after_original_idx']} 后插入")
                    print(f"    (Segment {p['segment_boundary'][0]} [ep={p['episode_index']}] -> Segment {p['segment_boundary'][1]} [ep={p['next_episode_index']}])")
        
        print("="*80 + "\n")
    
    def verify_placeholders(self, num_samples: int = 5):
        """
        验证占位符是否正确插入
        
        Args:
            num_samples: 检查的样本数量
        """
        print("\n" + "="*80)
        print("🔍 验证占位符")
        print("="*80)
        
        # 找出所有占位符的新索引
        placeholder_indices = [i for i, mapping in enumerate(self.new_to_original_idx) if mapping[1]]
        
        if not placeholder_indices:
            print("✓ 没有占位符需要验证")
            return
        
        print(f"总占位符数: {len(placeholder_indices)}")
        print(f"\n检查前 {min(num_samples, len(placeholder_indices))} 个占位符:\n")
        
        for i, placeholder_idx in enumerate(placeholder_indices[:num_samples]):
            frame = self[placeholder_idx]
            prev_frame = self[placeholder_idx - 1] if placeholder_idx > 0 else None
            next_frame = self[placeholder_idx + 1] if placeholder_idx < len(self) - 1 else None
            
            print(f"占位符 #{i+1} (新索引 {placeholder_idx}):")
            print(f"  is_placeholder: {frame['is_placeholder'].item()}")
            print(f"  action: {frame['action'][:3].tolist()}... (期望全为 {self.placeholder_value})")
            print(f"  episode_index: {frame['episode_index'].item()}")
            
            if prev_frame:
                print(f"  前一帧 (索引 {placeholder_idx-1}):")
                print(f"    is_placeholder: {prev_frame['is_placeholder'].item()}")
                print(f"    episode_index: {prev_frame['episode_index'].item()}")
                print(f"    action[:3]: {prev_frame['action'][:3].tolist()}")
            
            if next_frame:
                print(f"  后一帧 (索引 {placeholder_idx+1}):")
                print(f"    is_placeholder: {next_frame['is_placeholder'].item()}")
                print(f"    episode_index: {next_frame['episode_index'].item()}")
                print(f"    action[:3]: {next_frame['action'][:3].tolist()}")
            
            # 验证 action 是否全为占位符值
            is_valid = torch.all(frame['action'] == self.placeholder_value).item()
            print(f"  ✓ Action 验证: {'通过' if is_valid else '失败'}")
            print()
        
        print("="*80 + "\n")


def demo():
    """演示用法"""
    import sys
    
    # 示例：加载数据集
    dataset_path = "/home/dongyingyibadao/data_dealer_auto/cut_dataset"
    
    print("="*80)
    print("🚀 LeRobot Dataset with Placeholder - 演示")
    print("="*80 + "\n")
    
    # 1. 加载数据集
    dataset = LeRobotDatasetWithPlaceholder(
        repo_id='cut_dataset',
        root=dataset_path,
        placeholder_action_value=-999.0
    )
    
    # 2. 打印 episode 结构
    dataset.print_episode_structure()
    
    # 3. 验证占位符
    dataset.verify_placeholders(num_samples=3)
    
    # 4. 访问数据示例
    print("\n" + "="*80)
    print("📖 数据访问示例")
    print("="*80 + "\n")
    
    for i in range(min(10, len(dataset))):
        frame = dataset[i]
        placeholder_mark = "🔶 [PLACEHOLDER]" if frame['is_placeholder'].item() else ""
        action_str = f"[{frame['action'][0]:.2f}, {frame['action'][1]:.2f}, ...]" if not frame['is_placeholder'].item() else f"[{frame['action'][0]:.1f}, {frame['action'][1]:.1f}, ...]"
        
        print(f"索引 {i:3d}: Episode {frame['episode_index'].item():2d} | Action: {action_str} {placeholder_mark}")
    
    print("\n" + "="*80)
    print("✅ 演示完成")
    print("="*80)


if __name__ == '__main__':
    demo()
