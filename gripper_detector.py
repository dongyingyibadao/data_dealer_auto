"""
检测和识别抓取器夹爪状态变化的关键帧
"""
import torch
import numpy as np
from typing import List, Dict, Tuple


class GripperStateDetector:
    """
    检测夹爪状态变化（pick/place）的关键帧
    """
    
    def __init__(self, threshold: float = 0.5):
        """
        初始化抓取器状态检测器
        
        Args:
            threshold: 状态变化的阈值，默认为0.5
        """
        self.threshold = threshold
    
    def extract_gripper_state(self, action: torch.Tensor) -> float:
        """
        从动作向量中提取夹爪状态（第6个维度）
        
        Args:
            action: 动作向量 [x, y, z, α, β, γ, gripper]
            
        Returns:
            夹爪状态值 (-1.0 ~ 1.0)
        """
        if isinstance(action, torch.Tensor):
            return action[-1].item()
        else:
            return float(action[-1])
    
    def detect_gripper_changes(self, dataset, 
                               start_idx: int = 0,
                               end_idx: int = None) -> List[Dict]:
        """
        检测数据集中所有的夹爪状态变化
        
        Args:
            dataset: LeRobot数据集
            start_idx: 开始索引
            end_idx: 结束索引（None表示到末尾）
            
        Returns:
            关键帧信息列表，每项包含：
            {
                'index': 当前索引,
                'prev_gripper': 前一个夹爪状态,
                'curr_gripper': 当前夹爪状态,
                'action_type': 'pick' | 'place',
                'episode_index': episode索引,
                'frame_index': episode内帧索引
            }
        """
        if end_idx is None:
            end_idx = len(dataset)
        
        changes = []
        prev_gripper = None
        prev_idx = start_idx - 1
        
        print(f"🔍 开始检测夹爪状态变化 ({start_idx} - {end_idx})...")
        
        for i in range(start_idx, min(end_idx, len(dataset))):
            if i % 1000 == 0:
                print(f"  进度: {i}/{end_idx}")
            
            try:
                item = dataset[i]
                curr_gripper = self.extract_gripper_state(item['action'])
                
                # 检查是否存在状态变化
                if prev_gripper is not None:
                    diff = abs(curr_gripper - prev_gripper)
                    if diff > self.threshold:  # 状态发生显著变化
                        # 判断动作类型：-1.0 -> 1.0 是 pick，1.0 -> -1.0 是 place
                        if prev_gripper < 0 and curr_gripper > 0:
                            action_type = 'pick'
                        elif prev_gripper > 0 and curr_gripper < 0:
                            action_type = 'place'
                        else:
                            action_type = 'unknown'
                        
                        changes.append({
                            'index': i,
                            'prev_gripper': round(prev_gripper, 4),
                            'curr_gripper': round(curr_gripper, 4),
                            'action_type': action_type,
                            'episode_index': item.get('episode_index', -1),
                            'frame_index': item.get('frame_index', -1),
                            'task': item.get('task', 'unknown'),
                            'task_index': item.get('task_index', -1)
                        })
                
                prev_gripper = curr_gripper
                
            except Exception as e:
                print(f"⚠️  处理索引 {i} 时出错: {e}")
                continue
        
        print(f"✓ 检测完成，找到 {len(changes)} 个夹爪状态变化")
        return changes
    
    def extract_frame_ranges(self, 
                            dataset,
                            changes: List[Dict],
                            before_frames: int = 30,
                            after_frames: int = 30) -> List[Dict]:
        """
        从关键帧提取前后各N帧的范围
        
        Args:
            dataset: LeRobot数据集
            changes: 关键帧信息列表
            before_frames: 关键帧前取的帧数
            after_frames: 关键帧后取的帧数
            
        Returns:
            帧范围列表
        """
        ranges = []
        
        for change in changes:
            keyframe_idx = change['index']
            episode_idx = change['episode_index']
            frame_idx_in_episode = change['frame_index']
            
            # 计算当前episode的全局起始索引（假设frame_index是准确的且从0开始）
            # 如果frame_index不可用(-1)，则无法利用此优化，只能回溯查找
            if frame_idx_in_episode != -1:
                episode_start_global = keyframe_idx - frame_idx_in_episode
            else:
                # 回溯查找episode起点
                episode_start_global = keyframe_idx
                while episode_start_global > 0 and dataset[episode_start_global-1]['episode_index'] == episode_idx:
                    episode_start_global -= 1
            
            # 计算开始索引：不能早于episode开始
            start_idx = int(max(episode_start_global, keyframe_idx - before_frames))
            
            # 计算结束索引：不能晚于episode结束
            # 向后查找直到达到after_frames或episode结束
            end_idx = int(keyframe_idx + 1)
            frames_added = 0
            
            while frames_added < after_frames:
                if end_idx >= len(dataset):
                    break
                if dataset[end_idx]['episode_index'] != episode_idx:
                    break
                end_idx += 1
                frames_added += 1
            
            ranges.append({
                'keyframe_index': keyframe_idx,
                'action_type': change['action_type'],
                'frame_start': start_idx,
                'frame_end': end_idx,
                'num_frames': end_idx - start_idx,
                'episode_index': episode_idx,
                'frame_index': frame_idx_in_episode,
                'task': change['task'],
                'task_index': change['task_index'],
                'prev_gripper': change['prev_gripper'],
                'curr_gripper': change['curr_gripper']
            })
        
        return ranges
    
    def merge_adjacent_ranges(self, 
                             ranges: List[Dict],
                             min_gap: int = 50) -> List[Dict]:
        """
        合并相邻的帧范围（如果间隔过小）
        
        Args:
            ranges: 帧范围列表
            min_gap: 最小间隔阈值
            
        Returns:
            合并后的帧范围列表
        """
        if not ranges:
            return ranges
        
        merged = []
        current_range = ranges[0].copy()
        
        for next_range in ranges[1:]:
            # 检查是否是同一episode且间隔较小
            if (current_range['episode_index'] == next_range['episode_index'] and
                next_range['frame_start'] - current_range['frame_end'] < min_gap):
                # 合并范围
                current_range['frame_end'] = max(
                    current_range['frame_end'], 
                    next_range['frame_end']
                )
                current_range['num_frames'] = current_range['frame_end'] - current_range['frame_start']
            else:
                # 保存当前范围，开始新的
                merged.append(current_range)
                current_range = next_range.copy()
        
        merged.append(current_range)
        return merged


def analyze_gripper_changes(dataset, 
                           start_idx: int = 0,
                           end_idx: int = 10000,
                           before_frames: int = 30,
                           after_frames: int = 30,
                           merge: bool = False,
                           min_gap: int = 50
                           ) -> Tuple[List[Dict], List[Dict]]:
    """
    分析和提取夹爪状态变化
    
    Args:
        dataset: LeRobot数据集
        start_idx: 开始索引
        end_idx: 结束索引
        before_frames: 关键帧前取的帧数
        after_frames: 关键帧后取的帧数
        merge: 是否合并相邻范围
        min_gap: 合并的最小间隔阈值
        
    Returns:
        (changes, ranges) - 关键帧列表和帧范围列表
    """
    detector = GripperStateDetector(threshold=0.5)
    
    # 检测关键帧
    changes = detector.detect_gripper_changes(dataset, start_idx, end_idx)
    
    # 提取帧范围
    ranges = detector.extract_frame_ranges(dataset, changes, before_frames, after_frames)
    
    # 合并相邻范围
    if merge:
        merged_ranges = detector.merge_adjacent_ranges(ranges, min_gap)
    else:
        merged_ranges = ranges
    
    print(f"\n📊 分析结果:")
    print(f"  - 检测到的关键帧: {len(changes)}")
    print(f"  - 提取的帧范围: {len(ranges)}")
    print(f"  - 合并后的范围: {len(merged_ranges)}")
    
    # 统计pick/place比例
    pick_count = sum(1 for r in merged_ranges if r['action_type'] == 'pick')
    place_count = sum(1 for r in merged_ranges if r['action_type'] == 'place')
    print(f"  - Pick操作: {pick_count}")
    print(f"  - Place操作: {place_count}")
    
    return changes, merged_ranges


if __name__ == '__main__':
    print("Gripper Detector Module")
