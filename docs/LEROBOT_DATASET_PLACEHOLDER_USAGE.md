# LeRobot Dataset with Placeholder

为 motion_planning 系统提供的增强型 LeRobot Dataset 包装器。

## 🎯 功能特性

1. **自动插入占位符**：在同一原始 episode 的不同 segment 之间自动插入占位符帧
2. **跳跃标识**：占位符明确标记动作的跳跃边界，帮助机器人理解非连续动作
3. **完全兼容**：保持与原始 LeRobotDataset 的接口兼容
4. **透明访问**：通过标准索引访问，占位符自动处理

## 📋 占位符特性

每个占位符帧包含：

- `is_placeholder=True`：明确标记为占位符
- `action`：全为 `-999.0`（可配置的特殊值）
- `observation`：复制前一帧的观测数据
- `episode_index`：保持与所属 episode 相同
- `frame_index=-1`：无效的帧索引标记

## 🚀 快速开始

### 安装

```bash
# 确保已安装 lerobot
pip install lerobot torch
```

### 基本使用

```python
from lerobot_dataset_with_placeholder import LeRobotDatasetWithPlaceholder

# 加载数据集
dataset = LeRobotDatasetWithPlaceholder(
    repo_id='datasets_cut',
    root='/inspire/ssd/project/robot-decision/laijunxi-CZXS25230141/data_dealer_auto/datasets_cut',
    placeholder_action_value=-999.0  # 占位符的 action 值
)


# 查看数据集信息
print(f"总帧数: {len(dataset)} (包含占位符)")
print(f"原始帧数: {len(dataset.original_dataset)}")
print(f"占位符数: {dataset.num_placeholders}")

# 访问数据
for i in range(len(dataset)):
    frame = dataset[i]
    
    if frame['is_placeholder'].item():
        print(f"帧 {i}: 🔶 占位符 (episode {frame['episode_index'].item()})")
    else:
        print(f"帧 {i}: 正常帧 (episode {frame['episode_index'].item()})")
```

### 查看数据集结构

```python
# 打印所有 episode 的结构
dataset.print_episode_structure()

# 查看特定 episode
dataset.print_episode_structure(chunk_idx=0)

# 获取 episode 详细信息
info = dataset.get_episode_info(chunk_idx=0)
print(f"Segments: {info['num_segments']}")
print(f"Placeholders: {info['num_placeholders']}")
```

### 验证占位符

```python
# 验证占位符是否正确插入
dataset.verify_placeholders(num_samples=5)
```

## 📊 数据结构说明

### 原始数据集结构

当前 `cut_dataset` 的结构：

```
cut_dataset/
├── data/
│   ├── episode_0/              # 原始 Episode 0
│   │   ├── segment_0.parquet   # Segment 0 (episode_index=0)
│   │   ├── segment_1.parquet   # Segment 1 (episode_index=1)
│   │   ├── segment_2.parquet   # Segment 2 (episode_index=2)
│   │   └── segment_3.parquet   # Segment 3 (episode_index=3)
│   └── episode_1/              # 原始 Episode 1
│       ├── segment_4.parquet   # Segment 0 (episode_index=4)
│       ├── segment_5.parquet   # Segment 1 (episode_index=5)
│       ├── segment_6.parquet   # Segment 2 (episode_index=6)
│       ├── segment_7.parquet   # Segment 3 (episode_index=7)
│       ├── segment_8.parquet   # Segment 4 (episode_index=8)
│       └── segment_9.parquet   # Segment 5 (episode_index=9)
```

### 占位符插入位置

**原始 Episode 0** (4 个 segments → 3 个占位符):
```
Segment 0 (frames 0-25)
    ↓ [占位符 @ 新索引 26]
Segment 1 (frames 26-51)
    ↓ [占位符 @ 新索引 53]
Segment 2 (frames 52-77)
    ↓ [占位符 @ 新索引 80]
Segment 3 (frames 78-99)
```

**原始 Episode 1** (6 个 segments → 5 个占位符):
```
Segment 0 (frames 100-125)
    ↓ [占位符 @ 新索引 129]
Segment 1 (frames 126-151)
    ↓ [占位符 @ 新索引 156]
Segment 2 (frames 152-177)
    ↓ [占位符 @ 新索引 183]
Segment 3 (frames 178-203)
    ↓ [占位符 @ 新索引 210]
Segment 4 (frames 204-229)
    ↓ [占位符 @ 新索引 237]
Segment 5 (frames 230-255)
```

## 🔧 高级用法

### 在 motion_planning 中使用

```python
from lerobot_dataset_with_placeholder import LeRobotDatasetWithPlaceholder
import torch

dataset = LeRobotDatasetWithPlaceholder(
    repo_id='cut_dataset',
    root='./cut_dataset',
    placeholder_action_value=-999.0
)

# 训练循环
for idx in range(len(dataset)):
    frame = dataset[idx]
    
    if frame['is_placeholder'].item():
        # 占位符帧：重置或特殊处理
        print(f"检测到跳跃边界 @ 索引 {idx}")
        # 例如：重置轨迹缓冲区、保存当前轨迹片段等
        continue
    
    # 正常帧：处理观测和动作
    observation = {
        'image': frame['observation.images.image'],
        'state': frame['observation.state']
    }
    action = frame['action']
    
    # 你的训练逻辑...
```

### 过滤占位符

```python
# 只获取非占位符帧
real_frames = [
    dataset[i] 
    for i in range(len(dataset)) 
    if not dataset[i]['is_placeholder'].item()
]

print(f"真实帧数: {len(real_frames)}")
```

### 按 Episode 迭代

```python
# 迭代每个原始 episode
for chunk_idx in sorted(dataset.episode_segments.keys()):
    info = dataset.get_episode_info(chunk_idx)
    
    print(f"\n处理原始 Episode {chunk_idx}")
    print(f"  包含 {info['num_segments']} 个 segments")
    
    # 迭代该 episode 的所有 segment
    for seg in info['segments']:
        print(f"  Segment {seg['episode_index']}: {seg['length']} 帧")
        
        # 获取该 segment 的所有帧
        # 注意：需要将原始索引转换为新索引（考虑已插入的占位符）
```

## 📈 性能说明

- **内存开销**：占位符按需生成，不占用额外存储空间
- **访问速度**：单次访问 O(1)，与原始数据集相同
- **初始化时间**：额外分析 episode 结构，约增加 < 1 秒

## 🔍 调试和验证

### 运行演示脚本

```bash
cd /home/dongyingyibadao/data_dealer_auto
conda run -p /home/dongyingyibadao/miniconda3/envs/libero python lerobot_dataset_with_placeholder.py
```

### 检查特定帧

```python
# 查看 Segment 边界处的帧
for i in range(24, 29):  # Segment 0-1 边界附近
    frame = dataset[i]
    print(f"索引 {i}:")
    print(f"  episode_index: {frame['episode_index'].item()}")
    print(f"  is_placeholder: {frame['is_placeholder'].item()}")
    print(f"  action: {frame['action'][:3].tolist()}")
```

## ❓ 常见问题

### Q1: 为什么需要占位符？

**A**: 当一个完整的机器人任务被切分成多个片段（segments）时，相邻片段之间可能存在时间或动作的跳跃。占位符帮助 motion_planning 系统识别这些跳跃边界，避免模型错误地将非连续动作当作连续轨迹学习。

### Q2: 占位符会影响训练吗？

**A**: 不会。占位符有明确的 `is_placeholder=True` 标记和特殊的 action 值（-999），你可以在训练循环中跳过它们，或用于触发特殊逻辑（如轨迹片段的分割）。

### Q3: 如何自定义占位符的 action 值？

**A**: 在创建数据集时指定 `placeholder_action_value` 参数：

```python
dataset = LeRobotDatasetWithPlaceholder(
    repo_id='cut_dataset',
    root='./cut_dataset',
    placeholder_action_value=-1000.0  # 自定义值
)
```

### Q4: 不同 episode 之间会插入占位符吗？

**A**: 不会。占位符只在**同一原始 episode** 的不同 segments 之间插入。不同的原始 episode 之间保持独立，不插入占位符。

### Q5: 如何获取原始数据集？

**A**: 通过 `dataset.original_dataset` 访问：

```python
original_frame = dataset.original_dataset[100]  # 原始索引
```

## 📝 技术细节

### chunk_index vs episode_index

- **chunk_index**: 原始 episode 的索引（例如 episode_0, episode_1）
- **episode_index**: 切分后的 segment 索引（0, 1, 2, ...）

一个 chunk_index 可以对应多个 episode_index。例如：
- chunk_index=0 → episode_index=[0, 1, 2, 3]
- chunk_index=1 → episode_index=[4, 5, 6, 7, 8, 9]

### 索引映射

```python
# 新索引 -> (原始索引, is_placeholder, placeholder_info)
new_to_original_idx = [
    (0, False),       # 新索引 0 = 原始索引 0
    (1, False),       # 新索引 1 = 原始索引 1
    ...
    (25, False),      # 新索引 25 = 原始索引 25
    (-1, True, {...}),  # 新索引 26 = 占位符
    (26, False),      # 新索引 27 = 原始索引 26
    ...
]
```

## 🤝 与其他系统集成

### 与 PyTorch DataLoader 使用

```python
from torch.utils.data import DataLoader

dataset = LeRobotDatasetWithPlaceholder(
    repo_id='cut_dataset',
    root='./cut_dataset'
)

# 自定义 collate_fn 过滤占位符
def collate_fn(batch):
    # 过滤掉占位符
    batch = [item for item in batch if not item['is_placeholder'].item()]
    if not batch:
        return None
    # 标准的 batch 处理...
    return torch.utils.data.default_collate(batch)

loader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=collate_fn,
    shuffle=True
)
```

### 与 LeRobot 训练管道集成

```python
# 替换原始数据集
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot_dataset_with_placeholder import LeRobotDatasetWithPlaceholder

# 原来的代码
# dataset = LeRobotDataset('cut_dataset', root='./cut_dataset')

# 新代码
dataset = LeRobotDatasetWithPlaceholder(
    repo_id='cut_dataset',
    root='./cut_dataset'
)

# 其他代码保持不变
# dataset[i] 返回的数据格式完全相同，只是多了 is_placeholder 字段
```

## 📚 相关文档

- [LeRobot 官方文档](https://github.com/huggingface/lerobot)
- [Data Dealer Auto 使用指南](./README.md)
- [数据集格式修复总结](./FINAL_FIX_SUMMARY.md)

## 🐛 问题反馈

如果遇到问题，请检查：

1. ✅ 数据集路径正确
2. ✅ `meta/episodes/chunk-000/file-000.parquet` 存在
3. ✅ 数据集包含 `data/chunk_index` 字段
4. ✅ LeRobot 版本 >= 3.0

## 📊 示例输出

```
🔧 初始化带占位符的 LeRobot Dataset...
   repo_id: cut_dataset
   root: ./cut_dataset
🔍 分析 episode 结构...
   分析完成:
   - 原始 Episodes (chunk_index): 2
   - 切分后的 Segments: 10
   - 多 Segment Episodes: 2
   - 需插入占位符: 8 个
   原始 Episode 0 (chunk_index): 4 segments
      Segment 0 (episode_index=0): frames 0-25 (length=26)
      Segment 1 (episode_index=1): frames 26-51 (length=26)
      Segment 2 (episode_index=2): frames 52-77 (length=26)
      Segment 3 (episode_index=3): frames 78-99 (length=22)
   原始 Episode 1 (chunk_index): 6 segments
      Segment 0 (episode_index=4): frames 100-125 (length=26)
      ...
✅ 数据集加载完成
   原始帧数: 256
   新增占位符: 8
   总帧数: 264
   Episode数: 2
```

---

**版本**: 1.0.0  
**作者**: GitHub Copilot AI Assistant  
**日期**: 2024-12-09
