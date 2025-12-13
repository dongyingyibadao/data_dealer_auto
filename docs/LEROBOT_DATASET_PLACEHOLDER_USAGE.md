# LeRobot Dataset with Placeholder

为 motion_planning 系统提供的增强型 LeRobot Dataset 包装器。

## 🎯 功能特性

1. **自动插入占位符**：在同一原始 episode 的不同 segment 之间自动插入占位符帧
2. **跳跃标识**：占位符明确标记动作的跳跃边界，帮助机器人理解非连续动作
3. **Meta信息自动调整** ✨：`dataset.meta.episodes`中的`dataset_from_index`和`dataset_to_index`自动调整，与实际数据索引完全一致
4. **完全兼容**：保持与原始 LeRobotDataset 的接口兼容
5. **透明访问**：通过标准索引访问，占位符自动处理
6. **零文件修改**：纯内存操作，不修改任何磁盘文件

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


dataset = LeRobotDatasetWithPlaceholder(
    repo_id='HuggingFaceVLA_cus/datasets_cut',
    root='/inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA_cus/datasets_cut',
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

# 使用meta信息访问episode（索引已自动调整！）
for ep_idx in range(len(dataset.meta.episodes)):
    ep_meta = dataset.meta.episodes[ep_idx]
    from_idx = ep_meta['dataset_from_index']
    to_idx = ep_meta['dataset_to_index']
    
    # 直接使用meta中的索引 - 已考虑placeholder偏移
    first_frame = dataset[from_idx]
    last_frame = dataset[to_idx]
    
    print(f"Episode {ep_idx}: 范围 {from_idx}-{to_idx}, 任务: {ep_meta['tasks']}")
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

### 使用Meta信息（重要！）✨

**新特性**：`dataset.meta.episodes`中的索引已自动调整，可以直接使用！

```python
# Meta信息已自动调整，考虑了placeholder的偏移
ep_meta = dataset.meta.episodes[1]
from_idx = ep_meta['dataset_from_index']  # 已调整的索引
to_idx = ep_meta['dataset_to_index']      # 已调整的索引

# 直接使用，完全正确！
first_frame = dataset[from_idx]
last_frame = dataset[to_idx]

assert first_frame['episode_index'].item() == ep_meta['episode_index']
assert last_frame['episode_index'].item() == ep_meta['episode_index']

print(f"Episode {ep_meta['episode_index']}: 索引范围 {from_idx}-{to_idx}")
print(f"任务: {ep_meta['tasks']}")
```

如果需要访问原始的未调整meta：

```python
# 获取原始meta（未考虑placeholder偏移）
original_ep = dataset.original_meta.episodes[1]
original_from = original_ep['dataset_from_index']
original_to = original_ep['dataset_to_index']

# 比较
adjusted_ep = dataset.meta.episodes[1]
print(f"原始范围: {original_from}-{original_to}")
print(f"调整后: {adjusted_ep['dataset_from_index']}-{adjusted_ep['dataset_to_index']}")
print(f"偏移: +{adjusted_ep['dataset_from_index'] - original_from}")
```

**重要说明**：
- ✅ `dataset.meta`：返回调整后的meta（推荐使用）
- ✅ `dataset.original_meta`：返回原始meta（如需对比）
- ✅ 所有调整都在内存中完成，**不会修改磁盘文件**
- ✅ 完全透明，无需手动计算偏移量

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

### Q6: Meta信息的索引会自动调整吗？✨

**A**: 是的！从版本1.1开始，`dataset.meta.episodes`中的`dataset_from_index`和`dataset_to_index`会自动调整以考虑placeholder的偏移。你可以直接使用这些索引，无需手动计算：

```python
ep_meta = dataset.meta.episodes[1]
from_idx = ep_meta['dataset_from_index']  # 已自动调整

# 直接使用，完全正确
frame = dataset[from_idx]
```

如果需要原始的未调整索引，使用`dataset.original_meta`。

### Q7: 会修改原始的meta文件吗？

**A**: **不会！**所有meta调整都是纯内存操作：
- ❌ 不修改 `meta/info.json`
- ❌ 不修改 `meta/stats.json`
- ❌ 不修改 `meta/episodes/` 下的任何文件
- ✅ 只在内存中创建包装器，动态返回调整后的值
- ✅ 程序结束后，所有调整消失（因为只在内存中）

原始数据集文件完全安全，不会被修改。每次重新加载时，都会从原始文件读取。

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

### Meta信息动态调整机制✨

为了保持meta信息与实际数据索引一致，使用了**包装器模式**：

```python
class AdjustedEpisodesWrapper:
    """动态调整episode的dataset_from_index和dataset_to_index"""
    
    def __getitem__(self, idx):
        original_ep = self._original_episodes[idx]
        adjusted_ep = dict(original_ep)  # 创建副本，不修改原始数据
        
        # 应用索引偏移
        adjusted_ep['dataset_from_index'] = self._adjusted_ranges[idx]['dataset_from_index']
        adjusted_ep['dataset_to_index'] = self._adjusted_ranges[idx]['dataset_to_index']
        
        return adjusted_ep

class AdjustedMetadataWrapper:
    """包装原始meta，返回调整后的episodes"""
    
    @property
    def episodes(self):
        return self._adjusted_episodes  # 返回包装器
```

**工作流程**：
1. 加载数据集时，构建原始索引到新索引的映射表
2. 为每个episode计算调整后的`dataset_from_index`和`dataset_to_index`
3. 创建包装器对象，在访问时动态返回调整后的值
4. 原始meta文件保持不变（纯内存操作）

**优势**：
- ✅ 完全透明，使用方式与原始LeRobotDataset相同
- ✅ 自动调整，无需手动计算偏移
- ✅ 零文件修改，原始数据安全
- ✅ 惰性计算，不浪费内存

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

## 📚 更新日志

### v1.1.0 (2024-12-13) ✨
- ✅ 新增：Meta信息自动调整功能
- ✅ 新增：`dataset.meta.episodes`中的索引自动考虑placeholder偏移
- ✅ 新增：`dataset.original_meta`属性访问原始meta
- ✅ 改进：完全透明的使用体验，无需手动计算偏移
- ✅ 保证：纯内存操作，不修改任何磁盘文件

### v1.0.0 (2024-12-09)
- ✅ 初始版本：自动插入placeholder功能
- ✅ Episode结构分析和可视化
- ✅ Placeholder验证工具

---

**版本**: 1.1.0  
**作者**: GitHub Copilot AI Assistant  
**最后更新**: 2024-12-13
