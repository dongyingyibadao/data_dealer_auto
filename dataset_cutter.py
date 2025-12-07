"""
数据集裁剪和LeRobot格式转换
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
from datetime import datetime
import copy
from PIL import Image
import io


class DatasetCutter:
    """
    数据集裁剪器 - 提取指定范围的帧并支持两种保存模式：
    1. 图片模式：保存为图片文件（方便检查）
    2. LeRobot模式：保存为Parquet格式（方便训练）
    """
    
    def __init__(self, output_dir: str = None, save_mode: str = 'lerobot'):
        """
        初始化数据集裁剪器
        
        Args:
            output_dir: 输出目录
            save_mode: 保存模式 'image' 或 'lerobot' 或 'both'
        """
        self.output_dir = Path(output_dir) if output_dir else Path('./cut_dataset')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_mode = save_mode
        self.episodes_data = []
        self.metadata_buffer = []
    
    def extract_frames(self, 
                      dataset,
                      frame_ranges: List[Dict],
                      verbose: bool = True) -> List[Dict]:
        """
        从数据集中提取指定范围的帧
        
        Args:
            dataset: LeRobot数据集
            frame_ranges: 帧范围列表
            verbose: 是否打印详细信息
            
        Returns:
            提取的数据列表
        """
        extracted_data = []
        
        if verbose:
            print(f"📥 开始提取帧数据...")
        
        for range_idx, frame_range in enumerate(frame_ranges):
            if verbose and range_idx % 10 == 0:
                print(f"  处理范围 {range_idx}/{len(frame_ranges)}")
            
            start_idx = frame_range['frame_start']
            end_idx = frame_range['frame_end']
            
            for frame_idx in range(start_idx, end_idx):
                try:
                    item = dataset[frame_idx]
                    
                    # 复制数据项
                    new_item = copy.deepcopy({k: v for k, v in item.items() 
                                             if k in ['observation.images.image',
                                                     'observation.images.image2',
                                                     'observation.state',
                                                     'action',
                                                     'timestamp',
                                                     'frame_index',
                                                     'episode_index',
                                                     'task_index']})
                    
                    # 添加元数据
                    new_item['original_index'] = frame_idx
                    new_item['cut_range_id'] = range_idx
                    new_item['original_task'] = frame_range['task']
                    new_item['new_task'] = frame_range.get('new_task', frame_range['task'])
                    new_item['action_type'] = frame_range['action_type']
                    new_item['keyframe_index'] = frame_range['keyframe_index']
                    
                    extracted_data.append(new_item)
                
                except Exception as e:
                    if verbose:
                        print(f"⚠️  提取索引 {frame_idx} 时出错: {e}")
                    continue
        
        if verbose:
            print(f"✓ 提取完成，共 {len(extracted_data)} 帧")
        
        return extracted_data
    
    def organize_by_episode(self, 
                           extracted_data: List[Dict]) -> Dict[int, List[Dict]]:
        """
        按episode组织提取的数据
        
        Args:
            extracted_data: 提取的数据列表
            
        Returns:
            按episode_index组织的数据字典
        """
        episodes = {}
        
        for item in extracted_data:
            cut_range_id = item['cut_range_id']
            if cut_range_id not in episodes:
                episodes[cut_range_id] = {
                    'frames': [],
                    'metadata': {
                        'cut_range_id': cut_range_id,
                        'action_type': item['action_type'],
                        'original_task': item['original_task'],
                        'new_task': item['new_task'],
                        'episode_index': item.get('episode_index', -1),
                        'task_index': item.get('task_index', -1),
                        'keyframe_index': item['keyframe_index']
                    }
                }
            
            episodes[cut_range_id]['frames'].append(item)
        
        return episodes
    
    def save_as_image_format(self,
                           episodes_data: Dict[int, Dict],
                           frame_ranges: List[Dict],
                           max_episodes: Optional[int] = None) -> Path:
        """
        将数据保存为图片格式（类似data_dealer）
        
        Args:
            episodes_data: 按episode组织的数据
            frame_ranges: 帧范围列表
            max_episodes: 最多保存的episode数量
            
        Returns:
            保存的文件路径
        """
        print(f"💾 保存数据为图片格式...")
        
        # 创建输出目录
        images_dir = self.output_dir / 'images'
        images_dir.mkdir(parents=True, exist_ok=True)
        
        episodes_info = []
        
        for cut_range_id, episode_data in sorted(episodes_data.items()):
            if max_episodes and len(episodes_info) >= max_episodes:
                break
            
            frames = episode_data['frames']
            metadata = episode_data['metadata']
            
            episode_idx = len(episodes_info)
            episode_dir = images_dir / f"episode_{episode_idx:04d}"
            episode_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存每一帧的图像
            frame_files = []
            for frame_idx, frame_data in enumerate(frames):
                # 保存主摄像头图像
                img1 = self._tensor_to_image(frame_data['observation.images.image'])
                img1_path = episode_dir / f"frame_cam1_{frame_idx:04d}.jpg"
                img1.save(img1_path, quality=95)
                
                # 保存第二摄像头图像
                img2 = self._tensor_to_image(frame_data['observation.images.image2'])
                img2_path = episode_dir / f"frame_cam2_{frame_idx:04d}.jpg"
                img2.save(img2_path, quality=95)
                
                frame_files.append({
                    'frame_idx': frame_idx,
                    'cam1': str(img1_path.relative_to(self.output_dir)),
                    'cam2': str(img2_path.relative_to(self.output_dir)),
                    'action': frame_data['action'].cpu().numpy().tolist() if hasattr(frame_data['action'], 'cpu') else frame_data['action'].tolist(),
                    'state': frame_data['observation.state'].cpu().numpy().tolist() if hasattr(frame_data['observation.state'], 'cpu') else frame_data['observation.state'].tolist(),
                })
            
            episode_info = {
                'episode_idx': episode_idx,
                'cut_range_id': cut_range_id,
                'action_type': metadata['action_type'],
                'original_task': metadata['original_task'],
                'new_task': metadata['new_task'],
                'keyframe_index': metadata['keyframe_index'],
                'num_frames': len(frames),
                'frames': frame_files
            }
            
            episodes_info.append(episode_info)
            
            # 保存episode级别的元数据
            episode_meta_path = episode_dir / 'metadata.json'
            with open(episode_meta_path, 'w', encoding='utf-8') as f:
                json.dump(episode_info, f, indent=2, ensure_ascii=False)
        
        # 保存总体元数据
        summary_path = self.output_dir / 'episodes_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({
                'total_episodes': len(episodes_info),
                'episodes': episodes_info
            }, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ 保存了 {len(episodes_info)} 个episode的图片")
        print(f"  ✓ 元数据: {summary_path}")
        
        return self.output_dir
    
    @staticmethod
    def _tensor_to_image(tensor_data):
        """将Tensor转换为PIL Image"""
        if hasattr(tensor_data, 'cpu'):
            tensor_data = tensor_data.cpu()
        if hasattr(tensor_data, 'numpy'):
            tensor_data = tensor_data.numpy()
        
        # CHW -> HWC
        if tensor_data.ndim == 3 and tensor_data.shape[0] == 3:
            tensor_data = tensor_data.transpose(1, 2, 0)
        
        # 0-1 float -> 0-255 uint8
        if tensor_data.dtype != np.uint8:
            if tensor_data.max() <= 1.0:
                tensor_data = (tensor_data * 255).astype(np.uint8)
            else:
                tensor_data = tensor_data.astype(np.uint8)
        
        return Image.fromarray(tensor_data)
    
    def save_as_lerobot_format(self, 
                             episodes_data: Dict[int, Dict],
                             frame_ranges: List[Dict],
                             max_episodes: Optional[int] = None) -> Path:
        """
        将数据转换为LeRobot Parquet格式
        
        Args:
            episodes_data: 按episode组织的数据
            frame_ranges: 帧范围列表
            max_episodes: 最多保存的episode数量
            
        Returns:
            保存的文件路径
        """
        print(f"💾 保存数据为LeRobot Parquet格式...")
        
        # 创建输出目录结构
        meta_dir = self.output_dir / 'meta' / 'episodes' / 'chunk-000'
        # data_dir = self.output_dir / 'data' / 'chunk-000' # 不再使用单一的chunk目录
        data_root_dir = self.output_dir / 'data'
        meta_dir.mkdir(parents=True, exist_ok=True)
        data_root_dir.mkdir(parents=True, exist_ok=True)
        
        # 构建episodes元数据
        episodes_list = []
        global_frame_idx = 0
        
        # 保存帧数据
        print(f"\n  保存帧数据...")
        file_idx = 0
        
        for cut_range_id, episode_data in sorted(episodes_data.items()):
            if max_episodes and len(episodes_list) >= max_episodes:
                break
            
            frames = episode_data['frames']
            metadata = episode_data['metadata']
            
            num_frames = len(frames)
            
            # 当前新的episode index
            new_episode_idx = len(episodes_list)
            
            # 确保整数值不是Tensor
            def to_int(val):
                if hasattr(val, 'item'):
                    return int(val.item())
                return int(val)
            
            episode_meta = {
                'episode_index': new_episode_idx,
                'tasks': np.array([metadata['new_task']]),
                'dataset_from_index': global_frame_idx,
                'dataset_to_index': global_frame_idx + num_frames - 1,
                'length': num_frames,
                # 保留原始信息作为备注
                'action_type': metadata['action_type'],
                'original_task': metadata['original_task'],
                'cut_range_id': metadata['cut_range_id'],
                'keyframe_index': to_int(metadata['keyframe_index']),
                'original_episode_index': to_int(metadata['episode_index']),
                'original_task_index': to_int(metadata['task_index'])
            }
            
            episodes_list.append(episode_meta)
            global_frame_idx += num_frames
            
            # 准备帧数据
            frame_records = []
            for local_idx, frame in enumerate(frames):
                record = {
                    'observation.images.image': frame['observation.images.image'],
                    'observation.images.image2': frame['observation.images.image2'],
                    'observation.state': frame['observation.state'],
                    'action': frame['action'],
                    'timestamp': frame.get('timestamp', torch.tensor(0.0)),
                }
                frame_records.append(record)
            
            # 保存为parquet
            if frame_records:
                # 使用原始episode index创建文件夹
                original_ep_idx = to_int(metadata['episode_index'])
                episode_dir = data_root_dir / f'episode_{original_ep_idx}'
                episode_dir.mkdir(parents=True, exist_ok=True)
                
                # 文件名包含新的episode index以区分同一原始episode下的不同片段
                data_file = episode_dir / f'segment_{new_episode_idx}.parquet'
                
                self._save_frame_batch(frame_records, data_file)
                file_idx += 1
                
                if file_idx % 10 == 0:
                    print(f"    已保存 {file_idx} 个数据文件")

        # 转换为DataFrame
        episodes_df = pd.DataFrame(episodes_list)
        
        # 保存episodes元数据
        episodes_file = meta_dir / 'file-000.parquet'
        episodes_df.to_parquet(episodes_file, index=False)
        
        print(f"  ✓ 保存episodes元数据: {episodes_file}")
        print(f"    - Episodes数: {len(episodes_df)}")
        print(f"    - 总帧数: {global_frame_idx}")
        
        # 保存tasks列表 - 提取唯一的任务描述
        unique_tasks = set()
        for task_array in episodes_df['tasks']:
            task_str = task_array[0] if isinstance(task_array, np.ndarray) else task_array
            unique_tasks.add(task_str)
        
        tasks_data = []
        for task_idx, task in enumerate(sorted(unique_tasks)):
            tasks_data.append({'task_index': task_idx, 'task': task})
        
        tasks_df = pd.DataFrame(tasks_data)
        tasks_file = self.output_dir / 'meta' / 'tasks.parquet'
        tasks_df.to_parquet(tasks_file, index=False)
        
        print(f"  ✓ 保存tasks列表: {tasks_file}")
        print(f"    - Tasks数: {len(tasks_df)}")
        
        print(f"  ✓ 总共保存 {file_idx} 个数据文件")
        
        # 保存元信息
        self._save_metadata(meta_dir, episodes_df, tasks_df)
        
        return self.output_dir
    
    def _save_frame_batch(self, frame_records: List[Dict], file_path: Path):
        """
        保存一批帧数据为parquet文件，使用LeRobot的方式编码图像
        """
        from datasets import Dataset, Features, Image as HFImage, Sequence, Value
        
        def to_numpy(val):
            if hasattr(val, 'detach'):
                val = val.detach()
            if hasattr(val, 'cpu'):
                val = val.cpu()
            if hasattr(val, 'numpy'):
                return val.numpy()
            if isinstance(val, list):
                return np.array(val)
            return val
        
        # 将Tensor图像转换为PIL Image
        def tensor_to_pil(tensor_data):
            if hasattr(tensor_data, 'cpu'):
                tensor_data = tensor_data.cpu()
            if hasattr(tensor_data, 'numpy'):
                tensor_data = tensor_data.numpy()
            
            # CHW -> HWC
            if tensor_data.ndim == 3 and tensor_data.shape[0] == 3:
                tensor_data = tensor_data.transpose(1, 2, 0)
            
            # 0-1 float -> 0-255 uint8
            if tensor_data.dtype != np.uint8:
                if tensor_data.max() <= 1.0:
                    tensor_data = (tensor_data * 255).astype(np.uint8)
                else:
                    tensor_data = tensor_data.astype(np.uint8)
            
            return Image.fromarray(tensor_data)
        
        # 准备数据
        data = {
            'observation.images.image': [tensor_to_pil(f['observation.images.image']) for f in frame_records],
            'observation.images.image2': [tensor_to_pil(f['observation.images.image2']) for f in frame_records],
            'observation.state': [to_numpy(f['observation.state']).tolist() for f in frame_records],
            'action': [to_numpy(f['action']).tolist() for f in frame_records],
            'timestamp': [float(to_numpy(f['timestamp'])) for f in frame_records],
        }
        
        # 定义HuggingFace Dataset的Features
        features = Features({
            'observation.images.image': HFImage(),
            'observation.images.image2': HFImage(),
            'observation.state': Sequence(Value('float32')),
            'action': Sequence(Value('float32')),
            'timestamp': Value('float32'),
        })
        
        # 创建HuggingFace Dataset
        dataset = Dataset.from_dict(data, features=features)
        
        # 写入Parquet文件
        dataset.to_parquet(file_path)
    
    @staticmethod
    def _save_metadata(meta_dir: Path, episodes_df: pd.DataFrame, tasks_df: pd.DataFrame):
        """
        保存元信息文件
        """
        # 保存info.json
        info = {
            'total_episodes': len(episodes_df),
            'total_frames': episodes_df['length'].sum(),
            'total_tasks': len(tasks_df),
            'created_at': datetime.now().isoformat(),
            'robot_type': 'Panda 7-DOF',
            'observation_keys': ['observation.images.image', 'observation.images.image2', 'observation.state'],
            'action_keys': ['action'],
            'sampling_frequency': 10  # Hz
        }
        
        with open(meta_dir / 'info.json', 'w') as f:
            json.dump(info, f, indent=2, default=str)
        
        # 保存stats.json
        stats = {
            'total_episodes': int(len(episodes_df)),
            'total_frames': int(episodes_df['length'].sum()),
            'total_tasks': int(len(tasks_df)),
            'average_frames_per_episode': float(episodes_df['length'].mean()),
            'min_frames_per_episode': int(episodes_df['length'].min()),
            'max_frames_per_episode': int(episodes_df['length'].max()),
        }
        
        with open(meta_dir / 'stats.json', 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        print(f"  ✓ 保存元信息文件")


def cut_and_convert_dataset(dataset,
                           frame_ranges: List[Dict],
                           output_dir: str,
                           save_mode: str = 'lerobot',
                           max_episodes: Optional[int] = None) -> Path:
    """
    完整的数据集裁剪和转换流程
    
    Args:
        dataset: 原始LeRobot数据集
        frame_ranges: 帧范围列表（包含new_task字段）
        output_dir: 输出目录
        save_mode: 保存模式 'image'（图片）, 'lerobot'（Parquet）, 或 'both'（两者）
        max_episodes: 最多保存的episode数量
        
    Returns:
        输出目录路径
    """
    cutter = DatasetCutter(output_dir, save_mode=save_mode)
    
    # 提取帧
    extracted_data = cutter.extract_frames(dataset, frame_ranges)
    
    # 按episode组织
    episodes_data = cutter.organize_by_episode(extracted_data)
    
    # 根据模式保存
    if save_mode == 'image':
        output_path = cutter.save_as_image_format(episodes_data, frame_ranges, max_episodes)
    elif save_mode == 'lerobot':
        output_path = cutter.save_as_lerobot_format(episodes_data, frame_ranges, max_episodes)
    elif save_mode == 'both':
        print("\n📦 保存两种格式...\n")
        cutter.save_as_image_format(episodes_data, frame_ranges, max_episodes)
        output_path = cutter.save_as_lerobot_format(episodes_data, frame_ranges, max_episodes)
    else:
        raise ValueError(f"Unknown save_mode: {save_mode}. Use 'image', 'lerobot', or 'both'")
    
    return output_path


if __name__ == '__main__':
    print("Dataset Cutter Module")
