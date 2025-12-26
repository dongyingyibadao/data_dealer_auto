"""
数据集裁剪和LeRobot格式转换
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import json
from datetime import datetime
import copy
from PIL import Image
import io
import shutil
import os


class DatasetCutter:
    """
    数据集裁剪器 - 提取指定范围的帧并支持两种保存模式：
    1. 图片模式：保存为图片文件（方便检查）
    2. LeRobot模式：保存为Parquet格式（方便训练）
    """
    
    def __init__(self, output_dir: Optional[str] = None, save_mode: str = 'lerobot', batch_size: int = 100,
                 insert_placeholders: bool = False, placeholder_action_value: float = -999.0,
                 repo_id: Optional[str] = None, robot_type: str = "panda", fps: float = 10.0,
                 use_official_api: bool = True):
        """
        初始化数据集裁剪器
        
        Args:
            output_dir: 输出目录
            save_mode: 保存模式 'image' 或 'lerobot' 或 'both'
            batch_size: 批处理大小（每次处理多少个episode）
            insert_placeholders: 是否在同一chunk的不同segments之间物理插入placeholder（方案3）
            placeholder_action_value: placeholder的action值（默认-999.0）
            repo_id: HuggingFace repo ID（用于官方API）
            robot_type: 机器人类型（默认"panda"）
            fps: 采样频率（默认10.0）
            use_official_api: 是否使用LeRobot官方API（推荐）
        """
        self.output_dir = Path(output_dir) if output_dir else Path('./cut_dataset')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_mode = save_mode
        self.batch_size = batch_size
        self.insert_placeholders = insert_placeholders
        self.placeholder_action_value = placeholder_action_value
        self.use_official_api = use_official_api
        self.robot_type = robot_type
        self.fps = fps
        self.episodes_data = []
        self.metadata_buffer = []
        
        # 如果使用官方API，初始化LeRobotDataset
        self.lerobot_dataset = None
        if self.use_official_api and save_mode in ['lerobot', 'both']:
            # 清理已存在的数据集
            if repo_id is None:
                repo_id = f"custom/{self.output_dir.name}"
            self.repo_id = repo_id
            
            # 自动设置HF_LEROBOT_HOME为output_dir
            lerobot_home = self.output_dir.absolute()
            final_path = lerobot_home / repo_id
            print(f"  📍 设置数据保存路径: {final_path}/")
            
            # 使用官方API创建数据集
            try:
                # 方案：先导入模块，然后修改其中的HF_LEROBOT_HOME变量
                import lerobot.datasets.lerobot_dataset as lrd
                # 修改模块级别的变量
                lrd.HF_LEROBOT_HOME = lerobot_home
                # 保存到实例变量
                self._custom_lerobot_home = lerobot_home
                
                from lerobot.datasets.lerobot_dataset import LeRobotDataset
                
                # 清理已有数据集
                dataset_path = lerobot_home / repo_id
                if dataset_path.exists():
                    print(f"  ⚠️  清理已存在的数据集: {dataset_path}")
                    shutil.rmtree(dataset_path)
                
                print(f"  🔧 使用LeRobot官方API创建数据集: {repo_id}")
                self.lerobot_dataset = LeRobotDataset.create(
                    repo_id=repo_id,
                    robot_type=robot_type,
                    fps=int(fps),
                    features={
                        "observation.images.image": {
                            "dtype": "image",
                            "shape": (256, 256, 3),
                            "names": ["height", "width", "channel"],
                        },
                        "observation.images.image2": {
                            "dtype": "image",
                            "shape": (256, 256, 3),
                            "names": ["height", "width", "channel"],
                        },
                        "observation.state": {
                            "dtype": "float32",
                            "shape": (8,),
                            "names": ["state"],
                        },
                        "action": {
                            "dtype": "float32",
                            "shape": (7,),
                            "names": ["actions"],
                        },
                        "timestamp": {
                            "dtype": "float32",
                            "shape": (1,),
                            "names": None,
                        },
                        "frame_index": {
                            "dtype": "int64",
                            "shape": (1,),
                            "names": None,
                        },
                        "episode_index": {
                            "dtype": "int64",
                            "shape": (1,),
                            "names": None,
                        },
                        "index": {
                            "dtype": "int64",
                            "shape": (1,),
                            "names": None,
                        },
                        "task_index": {
                            "dtype": "int64",
                            "shape": (1,),
                            "names": None,
                        },
                        "is_last_segment":{
                            "dtype": "bool",
                            "shape": (1,),
                            "names": None,
                        }
                        
                    },
                    image_writer_threads=10,  # 并行优化
                    image_writer_processes=5,
                )
                print(f"  ✅ LeRobot数据集创建成功")
            except Exception as e:
                print(f"  ⚠️  LeRobot官方API初始化失败: {e}")
                print(f"  ℹ️  将使用传统方法保存数据")
                self.use_official_api = False
                self.lerobot_dataset = None
    
    def extract_frames_batch(self, 
                            dataset,
                            frame_ranges: List[Dict],
                            batch_start: int = 0,
                            batch_end: Optional[int] = None,
                            verbose: bool = True) -> List[Dict]:
        """
        从数据集中批量提取指定范围的帧（避免一次性加载所有数据）
        
        Args:
            dataset: LeRobot数据集
            frame_ranges: 帧范围列表
            batch_start: 批次起始索引
            batch_end: 批次结束索引（None表示到末尾）
            verbose: 是否打印详细信息
            
        Returns:
            提取的数据列表
        """
        extracted_data = []
        batch_end = batch_end or len(frame_ranges)
        
        if verbose:
            print(f"📥 提取帧数据批次 [{batch_start}:{batch_end}]...")
        
        for range_idx in range(batch_start, batch_end):
            if verbose and (range_idx - batch_start) % 10 == 0:
                print(f"  处理范围 {range_idx}/{batch_end}")
            
            frame_range = frame_ranges[range_idx]
            start_idx = frame_range['frame_start']
            end_idx = frame_range['frame_end']
            
            for frame_idx in range(start_idx, end_idx):
                try:
                    item = dataset[frame_idx]
                    
                    # 只提取需要的字段，不使用 deepcopy（内存优化）
                    new_item = {
                        'observation.images.image': item['observation.images.image'].clone().detach() if hasattr(item['observation.images.image'], 'clone') else item['observation.images.image'],
                        'observation.images.image2': item['observation.images.image2'].clone().detach() if hasattr(item['observation.images.image2'], 'clone') else item['observation.images.image2'],
                        'observation.state': item['observation.state'].clone().detach() if hasattr(item['observation.state'], 'clone') else item['observation.state'],
                        'action': item['action'].clone().detach() if hasattr(item['action'], 'clone') else item['action'],
                        'timestamp': item.get('timestamp', torch.tensor(0.0)),
                        'frame_index': item.get('frame_index', torch.tensor(0)),
                        'episode_index': item.get('episode_index', torch.tensor(0)),
                        'task_index': item.get('task_index', torch.tensor(0)),
                    }
                    
                    # 添加元数据
                    new_item['original_index'] = frame_idx
                    new_item['cut_range_id'] = range_idx
                    new_item['original_task'] = frame_range.get('original_task', frame_range.get('task', ''))
                    new_item['new_task'] = frame_range.get('new_task', frame_range.get('original_task', frame_range.get('task', '')))
                    new_item['action_type'] = frame_range['action_type']
                    new_item['keyframe_index'] = frame_range['keyframe_index']
                    
                    extracted_data.append(new_item)
                
                except Exception as e:
                    if verbose:
                        print(f"⚠️  提取索引 {frame_idx} 时出错: {e}")
                    continue
        
        if verbose:
            print(f"✓ 批次提取完成，共 {len(extracted_data)} 帧")
        
        return extracted_data
    
    def organize_by_episode(self, 
                           extracted_data: List[Dict]) -> Dict[int, Dict]:
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
    
    def _create_placeholder_frame(self, previous_frame: Dict, episode_index: int, 
                                  global_frame_idx: int, task_index: int) -> Dict:
        """
        创建一个placeholder帧（方案3：物理写入）
        
        Args:
            previous_frame: 前一帧的数据（用于复制observation）
            episode_index: 当前episode索引
            global_frame_idx: 全局帧索引
            task_index: 任务索引
            
        Returns:
            placeholder帧数据
        """
        # 复制observation（图像和状态）
        placeholder = {
            'observation.images.image': previous_frame['observation.images.image'].clone(),
            'observation.images.image2': previous_frame['observation.images.image2'].clone(),
            'observation.state': previous_frame['observation.state'].clone(),
        }
        
        # 设置特殊的action值（全为placeholder_action_value）
        action_shape = previous_frame['action'].shape
        placeholder['action'] = torch.full(action_shape, self.placeholder_action_value, 
                                          dtype=previous_frame['action'].dtype)
        
        # 设置元数据
        placeholder['timestamp'] = previous_frame.get('timestamp', torch.tensor(0.0))
        placeholder['episode_index'] = torch.tensor(episode_index)
        placeholder['frame_index'] = torch.tensor(-1)  # 特殊标记
        placeholder['index'] = torch.tensor(global_frame_idx)
        placeholder['task_index'] = torch.tensor(task_index)
        
        return placeholder
    
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
    
    @staticmethod
    def _tensor_to_numpy_image(tensor_data):
        """将Tensor转换为numpy图像（用于LeRobot API）"""
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
        
        return tensor_data
    
    @staticmethod
    def _tensor_to_numpy(tensor_data):
        """将Tensor转换为numpy数组"""
        if hasattr(tensor_data, 'cpu'):
            tensor_data = tensor_data.cpu()
        if hasattr(tensor_data, 'numpy'):
            return tensor_data.numpy()
        return np.array(tensor_data)
    
    def save_as_lerobot_format_streaming(self,
                                        dataset,
                                        frame_ranges: List[Dict],
                                        max_episodes: Optional[int] = None) -> Path:
        """
        流式保存数据为LeRobot格式（批处理，节省内存）
        支持使用官方API或传统方法
        
        Args:
            dataset: 原始LeRobot数据集
            frame_ranges: 帧范围列表
            max_episodes: 最多保存的episode数量
            
        Returns:
            保存的文件路径
        """
        # 如果使用官方API
        if self.use_official_api and self.lerobot_dataset is not None:
            return self._save_with_official_api(dataset, frame_ranges, max_episodes)
        else:
            # 使用传统方法
            return self._save_with_traditional_method(dataset, frame_ranges, max_episodes)
    
    def _save_with_official_api(self,
                                dataset,
                                frame_ranges: List[Dict],
                                max_episodes: Optional[int] = None) -> Path:
        """
        使用LeRobot官方API保存数据
        
        Args:
            dataset: 原始LeRobot数据集
            frame_ranges: 帧范围列表
            max_episodes: 最多保存的episode数量
            
        Returns:
            保存的文件路径
        """
        print(f"💾 使用LeRobot官方API保存数据...")
        print(f"  批处理大小: {self.batch_size} episodes/批")

        if self.lerobot_dataset is None:
            raise RuntimeError("LeRobot dataset 未初始化")
        lrd = self.lerobot_dataset
        
        # 限制episode数量
        total_ranges = min(len(frame_ranges), max_episodes) if max_episodes else len(frame_ranges)
        
        # 分批处理
        for batch_start in range(0, total_ranges, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_ranges)
            
            print(f"\n  处理批次 [{batch_start}:{batch_end}]/{total_ranges}")
            
            # 提取当前批次的帧数据
            extracted_data = self.extract_frames_batch(
                dataset, frame_ranges, batch_start, batch_end, verbose=False
            )
            
            # 按episode组织
            episodes_data = self.organize_by_episode(extracted_data)
            
            # 处理每个episode
            # 用于缓存下一个episode需要的placeholder
            # pending_placeholder = None
            
            for cut_range_id, episode_data in sorted(episodes_data.items()):
                frames = episode_data['frames']
                metadata = episode_data['metadata']
                task_name = metadata['new_task']
                
                
                # 判断该segment是否为原始episode的最后一个片段
                is_last_segment = False
                next_idx = cut_range_id + 1
                if next_idx < len(frame_ranges):
                    next_metadata = frame_ranges[next_idx]
                    if next_metadata.get('episode_index', -1) != metadata['episode_index']:
                        is_last_segment = True
                else:
                    is_last_segment= True
                
                is_last_segment = np.array([is_last_segment])
                # 使用官方API逐帧添加
                for frame_idx, frame in enumerate(frames):
                    # 转换图像格式（LeRobot API需要numpy格式）
                    image1 = self._tensor_to_numpy_image(frame['observation.images.image'])
                    image2 = self._tensor_to_numpy_image(frame['observation.images.image2'])
                    state = self._tensor_to_numpy(frame['observation.state'])
                    action = self._tensor_to_numpy(frame['action'])
                    
                    # 注意：timestamp, frame_index, episode_index, index, task_index
                    # 这些字段由官方API自动生成，不需要手动传入
                    lrd.add_frame({
                        "observation.images.image": image1,
                        "observation.images.image2": image2,
                        "observation.state": state,
                        "action": action,
                        "task": task_name,
                        "is_last_segment": is_last_segment,
                    })
                
                if self.insert_placeholders:
                    placeholder_action = np.full((7,), self.placeholder_action_value, dtype=np.float32)
                    last_frame = frames[-1]
                    image1 = self._tensor_to_numpy_image(last_frame['observation.images.image'])
                    image2 = self._tensor_to_numpy_image(last_frame['observation.images.image2'])
                    state = self._tensor_to_numpy(last_frame['observation.state'])
                    

                    lrd.add_frame({
                        "observation.images.image": image1,
                        "observation.images.image2": image2,
                        "observation.state": state,
                        "action": placeholder_action,
                        "task": task_name,
                        "is_last_segment": is_last_segment,
                    })
                
                
                # # 检查是否需要为当前episode末尾准备placeholder
                # if self.insert_placeholders:
                #     next_idx = cut_range_id + 1
                #     if next_idx < len(frame_ranges):
                #         next_metadata = frame_ranges[next_idx]
                #         if next_metadata.get('episode_index', -1) == metadata['episode_index']:
                #             # 同一个chunk，准备placeholder（将在当前episode末尾插入）
                #             last_frame = frames[-1]
                #             image1 = self._tensor_to_numpy_image(last_frame['observation.images.image'])
                #             image2 = self._tensor_to_numpy_image(last_frame['observation.images.image2'])
                #             state = self._tensor_to_numpy(last_frame['observation.state'])
                            
                #             # Placeholder action全为特殊值
                #             placeholder_action = np.full((7,), self.placeholder_action_value, dtype=np.float32)
                            
                #             # 准备placeholder数据
                #             pending_placeholder = {
                #                 "observation.images.image": image1,
                #                 "observation.images.image2": image2,
                #                 "observation.state": state,
                #                 "action": placeholder_action,
                #                 # "task": f"[PLACEHOLDER] {task_name}→{next_metadata.get('new_task', '')}",
                #                 "task": task_name,
                #             }
                
                # 保存episode（不包含placeholder）
                lrd.save_episode()
            
            print(f"  ✓ 批次完成，已保存 {len(episodes_data)} episodes")
            
            # 清理内存
            del extracted_data
            del episodes_data
            import gc
            gc.collect()
        
        print(f"\n✅ 使用官方API保存完成!")
        print(f"  总episodes: {total_ranges}")
        
        # 返回数据集路径（使用我们自定义的路径）
        return self._custom_lerobot_home / self.repo_id
    
    def _save_with_traditional_method(self,
                                     dataset,
                                     frame_ranges: List[Dict],
                                     max_episodes: Optional[int] = None) -> Path:
        """
        使用传统方法保存数据（向后兼容）
        
        Args:
            dataset: 原始LeRobot数据集
            frame_ranges: 帧范围列表
            max_episodes: 最多保存的episode数量
            
        Returns:
            保存的文件路径
        """
        print(f"💾 使用传统方法保存数据...")
        print(f"  批处理大小: {self.batch_size} episodes/批")
        
        # 首先构建任务映射表
        task_to_index = {}
        for frame_range in frame_ranges:
            task_desc = frame_range.get('new_task', frame_range.get('task', ''))
            if task_desc not in task_to_index:
                task_to_index[task_desc] = len(task_to_index)
        
        print(f"\n  任务映射表:")
        for task, idx in sorted(task_to_index.items(), key=lambda x: x[1]):
            print(f"    {idx}: {task}")
        
        # 创建输出目录结构
        meta_dir = self.output_dir / 'meta' / 'episodes' / 'chunk-000'
        data_root_dir = self.output_dir / 'data'
        meta_dir.mkdir(parents=True, exist_ok=True)
        data_root_dir.mkdir(parents=True, exist_ok=True)
        
        # 流式处理
        episodes_list = []
        global_frame_idx = 0
        file_idx = 0
        
        # 限制episode数量
        total_ranges = min(len(frame_ranges), max_episodes) if max_episodes else len(frame_ranges)
        
        # 分批处理
        for batch_start in range(0, total_ranges, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_ranges)
            
            print(f"\n  处理批次 [{batch_start}:{batch_end}]/{total_ranges}")
            
            # 提取当前批次的帧数据
            extracted_data = self.extract_frames_batch(
                dataset, frame_ranges, batch_start, batch_end, verbose=False
            )
            
            # 按episode组织
            episodes_data = self.organize_by_episode(extracted_data)
            
            # 保存当前批次
            for cut_range_id, episode_data in sorted(episodes_data.items()):
                frames = episode_data['frames']
                metadata = episode_data['metadata']
                
                num_frames = len(frames)
                new_episode_idx = len(episodes_list)
                
                # 确保整数值不是Tensor
                def to_int(val):
                    if hasattr(val, 'item'):
                        return int(val.item())
                    return int(val)
                
                episode_meta = {
                    'episode_index': new_episode_idx,
                    'tasks': np.array([metadata['new_task']]),
                    'data/chunk_index': to_int(metadata['episode_index']),
                    'data/file_index': new_episode_idx,
                    'dataset_from_index': global_frame_idx,
                    'dataset_to_index': global_frame_idx + num_frames - 1,
                    'length': num_frames,
                    'action_type': metadata['action_type'],
                    'original_task': metadata['original_task'],
                    'cut_range_id': metadata['cut_range_id'],
                    'keyframe_index': to_int(metadata['keyframe_index']),
                    'original_episode_index': to_int(metadata['episode_index']),
                    'original_task_index': to_int(metadata['task_index'])
                }
                
                episodes_list.append(episode_meta)
                global_frame_idx += num_frames
                
                # 获取当前episode的task_index
                current_task = metadata['new_task']
                current_task_index = task_to_index[current_task]
                
                # 准备帧数据
                frame_records = []
                for local_idx, frame in enumerate(frames):
                    record = {
                        'observation.images.image': frame['observation.images.image'],
                        'observation.images.image2': frame['observation.images.image2'],
                        'observation.state': frame['observation.state'],
                        'action': frame['action'],
                        'timestamp': frame.get('timestamp', torch.tensor(0.0)),
                        'episode_index': torch.tensor(new_episode_idx),
                        'frame_index': torch.tensor(local_idx),
                        'index': torch.tensor(global_frame_idx - num_frames + local_idx),
                        'task_index': torch.tensor(current_task_index),
                    }
                    frame_records.append(record)
                
                # 插入placeholder（如果启用且不是最后一个episode）
                placeholder_added = False
                if self.insert_placeholders and new_episode_idx < total_ranges - 1:
                    # 检查下一个episode是否属于同一个chunk
                    next_idx = cut_range_id + 1
                    if next_idx < len(frame_ranges):
                        next_metadata = frame_ranges[next_idx]
                        if next_metadata.get('episode_index', -1) == metadata['episode_index']:
                            # 同一个chunk，将placeholder追加到当前segment
                            placeholder_frame_dict = self._create_placeholder_frame(
                                frames[-1],  # 使用当前segment的最后一帧
                                new_episode_idx,
                                global_frame_idx,  # placeholder使用下一个frame的index
                                current_task_index
                            )
                            
                            # 将placeholder作为额外帧追加到frame_records
                            frame_records.append(placeholder_frame_dict)
                            global_frame_idx += 1  # placeholder占用一个frame
                            placeholder_added = True
                            
                            if new_episode_idx < 3:  # 只打印前几个
                                print(f"  ⚡ 插入placeholder @ 索引 {global_frame_idx-1} (追加到 segment {new_episode_idx})")
                
                # 保存为parquet（包含可能的placeholder帧）
                if frame_records:
                    original_ep_idx = to_int(metadata['episode_index'])
                    episode_dir = data_root_dir / f'episode_{original_ep_idx}'
                    episode_dir.mkdir(parents=True, exist_ok=True)
                    
                    data_file = episode_dir / f'segment_{new_episode_idx}.parquet'
                    self._save_frame_batch(frame_records, data_file)
                    file_idx += 1
                
                # 调整episode metadata以包含placeholder
                if placeholder_added:
                    episode_meta['length'] += 1  # 增加1帧（placeholder）
                    episode_meta['dataset_to_index'] += 1  # 结束索引后移
            
            # 清理内存
            del extracted_data
            del episodes_data
            import gc
            gc.collect()
            
            print(f"  ✓ 批次完成，已处理 {len(episodes_list)} episodes, {file_idx} 文件")
        
        # 保存元数据
        episodes_df = pd.DataFrame(episodes_list)
        episodes_file = meta_dir / 'file-000.parquet'
        episodes_df.to_parquet(episodes_file, index=False)
        
        print(f"\n  ✓ 保存episodes元数据: {episodes_file}")
        print(f"    - Episodes数: {len(episodes_df)}")
        print(f"    - 总帧数: {global_frame_idx}")
        
        # 保存tasks列表
        tasks_data = []
        for task, task_idx in sorted(task_to_index.items(), key=lambda x: x[1]):
            tasks_data.append({'task': task, 'task_index': task_idx})
        
        tasks_df = pd.DataFrame(tasks_data)
        tasks_df = tasks_df.set_index('task')
        tasks_file = self.output_dir / 'meta' / 'tasks.parquet'
        tasks_df.to_parquet(tasks_file, index=True)
        
        print(f"  ✓ 保存tasks列表: {tasks_file}")
        print(f"    - Tasks数: {len(tasks_df)}")
        print(f"  ✓ 总共保存 {file_idx} 个数据文件")
        
        # 保存元信息
        root_meta_dir = self.output_dir / 'meta'
        self._save_metadata(root_meta_dir, episodes_df, tasks_df)
        
        return self.output_dir
    
    def save_as_lerobot_format(self, 
                             episodes_data: Dict[int, Dict],
                             frame_ranges: List[Dict],
                             max_episodes: Optional[int] = None) -> Path:
        """
        将数据转换为LeRobot Parquet格式（旧版本，保留兼容性）
        
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
        
        # 首先收集所有唯一的任务描述，构建任务索引映射
        task_to_index = {}
        for cut_range_id, episode_data in sorted(episodes_data.items()):
            task_desc = episode_data['metadata']['new_task']
            if task_desc not in task_to_index:
                task_to_index[task_desc] = len(task_to_index)
        
        print(f"\n  任务映射表:")
        for task, idx in sorted(task_to_index.items(), key=lambda x: x[1]):
            print(f"    {idx}: {task}")
        
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
                # LeRobot required: data file location
                'data/chunk_index': to_int(metadata['episode_index']),  # 使用原始episode作为chunk
                'data/file_index': new_episode_idx,  # 使用新episode index作为file index
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
            
            # 获取当前episode的task_index
            current_task = metadata['new_task']
            current_task_index = task_to_index[current_task]
            
            # 准备帧数据
            frame_records = []
            for local_idx, frame in enumerate(frames):
                record = {
                    'observation.images.image': frame['observation.images.image'],
                    'observation.images.image2': frame['observation.images.image2'],
                    'observation.state': frame['observation.state'],
                    'action': frame['action'],
                    'timestamp': frame.get('timestamp', torch.tensor(0.0)),
                    # 添加必需的元数据字段 - 强制使用新的索引值
                    'episode_index': torch.tensor(new_episode_idx),  # 使用新episode索引
                    'frame_index': torch.tensor(local_idx),  # 使用局部帧索引
                    'index': torch.tensor(global_frame_idx - num_frames + local_idx),  # 全局索引
                    'task_index': torch.tensor(current_task_index),  # 使用正确的任务索引
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
        
        # 保存tasks列表 - 使用预先构建的任务映射
        # 注意：任务描述应该作为DataFrame的index（行名），task_index作为列
        tasks_data = []
        for task, task_idx in sorted(task_to_index.items(), key=lambda x: x[1]):
            tasks_data.append({'task': task, 'task_index': task_idx})
        
        tasks_df = pd.DataFrame(tasks_data)
        # 将任务描述设为index（这是LeRobot期望的格式）
        tasks_df = tasks_df.set_index('task')
        tasks_file = self.output_dir / 'meta' / 'tasks.parquet'
        tasks_df.to_parquet(tasks_file, index=True)  # 确保保存index
        
        print(f"  ✓ 保存tasks列表: {tasks_file}")
        print(f"    - Tasks数: {len(tasks_df)}")
        
        print(f"  ✓ 总共保存 {file_idx} 个数据文件")
        
        # 保存元信息 - 传递正确的meta根目录
        root_meta_dir = self.output_dir / 'meta'
        self._save_metadata(root_meta_dir, episodes_df, tasks_df)
        
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
            # 添加元数据字段
            'episode_index': [int(to_numpy(f['episode_index'])) for f in frame_records],
            'frame_index': [int(to_numpy(f['frame_index'])) for f in frame_records],
            'index': [int(to_numpy(f['index'])) for f in frame_records],
            'task_index': [int(to_numpy(f['task_index'])) for f in frame_records],
        }
        
        # 定义HuggingFace Dataset的Features
        features = Features({
            'observation.images.image': HFImage(),
            'observation.images.image2': HFImage(),
            'observation.state': Sequence(Value('float32')),
            'action': Sequence(Value('float32')),
            'timestamp': Value('float32'),
            # 添加元数据字段
            'episode_index': Value('int64'),
            'frame_index': Value('int64'),
            'index': Value('int64'),
            'task_index': Value('int64'),
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
        print(f"  📝 开始保存元信息到 {meta_dir}...")
        
        # 保存info.json - 完整的LeRobot格式
        info = {
            'codebase_version': 'v3.0',
            'robot_type': 'Panda 7-DOF',
            'total_episodes': int(len(episodes_df)),
            'total_frames': int(episodes_df['length'].sum()),
            'total_tasks': int(len(tasks_df)),
            'chunks_size': 1000,
            'fps': 10.0,
            'splits': {
                'train': f"0:{len(episodes_df)}"
            },
            'data_path': 'data/episode_{chunk_index}/segment_{file_index}.parquet',
            'features': {
                'observation.images.image': {
                    'dtype': 'image',
                    'shape': [256, 256, 3],
                    'names': ['height', 'width', 'channel'],
                    'fps': 10.0
                },
                'observation.images.image2': {
                    'dtype': 'image',
                    'shape': [256, 256, 3],
                    'names': ['height', 'width', 'channel'],
                    'fps': 10.0
                },
                'observation.state': {
                    'dtype': 'float32',
                    'shape': [8],
                    'names': ['state'],
                    'fps': 10.0
                },
                'action': {
                    'dtype': 'float32',
                    'shape': [7],
                    'names': ['actions'],
                    'fps': 10.0
                },
                'timestamp': {
                    'dtype': 'float32',
                    'shape': [1],
                    'names': None,
                    'fps': 10.0
                },
                'episode_index': {
                    'dtype': 'int64',
                    'shape': [1],
                    'names': None,
                    'fps': 10.0
                },
                'frame_index': {
                    'dtype': 'int64',
                    'shape': [1],
                    'names': None,
                    'fps': 10.0
                },
                'index': {
                    'dtype': 'int64',
                    'shape': [1],
                    'names': None,
                    'fps': 10.0
                },
                'task_index': {
                    'dtype': 'int64',
                    'shape': [1],
                    'names': None,
                    'fps': 10.0
                }
            }
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
                           output_dir: Optional[str],
                           save_mode: str = 'lerobot',
                           max_episodes: Optional[int] = None,
                           batch_size: int = 100,
                           streaming: bool = True,
                           insert_placeholders: bool = False,
                           placeholder_action_value: float = -999.0,
                           repo_id: Optional[str] = None,
                           robot_type: str = "panda",
                           fps: float = 10.0,
                           use_official_api: bool = True) -> Path:
    """
    完整的数据集裁剪和转换流程
    
    Args:
        dataset: 原始LeRobot数据集
        frame_ranges: 帧范围列表（包含new_task字段）
        output_dir: 输出目录
        save_mode: 保存模式 'image'（图片）, 'lerobot'（Parquet）, 或 'both'（两者）
        max_episodes: 最多保存的episode数量
        batch_size: 批处理大小（每次处理多少个episode）
        streaming: 是否使用流式处理（推荐，节省内存）
        insert_placeholders: 是否在同一chunk的不同segments之间物理插入placeholder（方案3）
        placeholder_action_value: placeholder的action值（默认-999.0）
        repo_id: HuggingFace repo ID（用于官方API）
        robot_type: 机器人类型（默认"panda"）
        fps: 采样频率（默认10.0）
        use_official_api: 是否使用LeRobot官方API（推荐）
        
    Returns:
        输出目录路径
    """
    cutter = DatasetCutter(output_dir, save_mode=save_mode, batch_size=batch_size,
                          insert_placeholders=insert_placeholders,
                          placeholder_action_value=placeholder_action_value,
                          repo_id=repo_id, robot_type=robot_type, fps=fps,
                          use_official_api=use_official_api)
    
    # 使用流式处理（推荐）
    if streaming and save_mode in ['lerobot', 'both']:
        print(f"\n💡 使用流式处理模式（批大小: {batch_size}）")
        output_path = cutter.save_as_lerobot_format_streaming(dataset, frame_ranges, max_episodes)
        
        # 如果需要同时保存图片格式
        if save_mode == 'both':
            print("\n📦 额外保存图片格式...\n")
            # 图片格式也使用批处理
            for batch_start in range(0, len(frame_ranges), batch_size):
                batch_end = min(batch_start + batch_size, len(frame_ranges))
                extracted_data = cutter.extract_frames_batch(dataset, frame_ranges, batch_start, batch_end)
                episodes_data = cutter.organize_by_episode(extracted_data)
                cutter.save_as_image_format(episodes_data, frame_ranges[batch_start:batch_end], max_episodes)
                del extracted_data, episodes_data
                import gc
                gc.collect()
    else:
        # 旧方式：一次性加载所有数据（不推荐，但保留兼容性）
        print(f"\n⚠️  使用传统处理模式（一次性加载所有数据）")
        extracted_data = cutter.extract_frames_batch(dataset, frame_ranges, 0, len(frame_ranges))
        episodes_data = cutter.organize_by_episode(extracted_data)
        
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
