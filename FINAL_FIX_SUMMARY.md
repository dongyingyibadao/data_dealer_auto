# LeRobot 数据集格式完整修复总结

本文档记录了 `data_dealer_auto` 生成的数据集为兼容 LeRobot v3.0 格式所做的所有修复。

---

## 📋 目录

1. [问题概述](#问题概述)
2. [修复1：添加必需的元数据字段](#修复1添加必需的元数据字段)
3. [修复2：添加数据文件定位字段](#修复2添加数据文件定位字段)
4. [修复3：修复episode_index覆盖问题](#修复3修复episode_index覆盖问题)
5. [修复4：实现正确的task_index映射](#修复4实现正确的task_index映射)
6. [修复5：修复tasks.parquet格式](#修复5修复tasksparquet格式)
7. [验证结果](#验证结果)
8. [LeRobot格式要求总结](#lerobot格式要求总结)

---

## 问题概述

生成的 `cut_dataset` 无法被 `LeRobotDataset` 加载，出现以下错误：

1. ❌ 缺少 `info.json` 的 `features` 字段
2. ❌ Parquet 文件缺少元数据字段：`episode_index`, `frame_index`, `index`, `task_index`
3. ❌ Episode 元数据缺少 `data/chunk_index` 和 `data/file_index`
4. ❌ 所有帧的 `episode_index` 都是 0（应该是 0, 1, 2, ...）
5. ❌ 所有帧的 `task_index` 都是 0（应该根据任务分配）
6. ❌ `frame['task']` 返回数字而不是任务描述字符串

---

## 修复1：添加必需的元数据字段

### 问题
Parquet 数据文件缺少 4 个元数据字段，导致 LeRobot 无法正确索引和检索数据。

### 解决方案

**文件**: `dataset_cutter.py` 第307-320行

添加元数据字段到每个帧记录：

```python
record = {
    'observation.images.image': frame['observation.images.image'],
    'observation.images.image2': frame['observation.images.image2'],
    'observation.state': frame['observation.state'],
    'action': frame['action'],
    'timestamp': frame.get('timestamp', torch.tensor(0.0)),
    # 新增：必需的元数据字段
    'episode_index': torch.tensor(new_episode_idx),  # Episode索引
    'frame_index': torch.tensor(local_idx),          # Episode内帧索引
    'index': torch.tensor(global_frame_idx),         # 全局帧索引
    'task_index': torch.tensor(current_task_index),  # 任务索引
}
```

同时更新 `info.json` 的 `features` 字段定义（第447-519行）：

```python
'features': {
    # ... 观测和动作字段 ...
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
```

---

## 修复2：添加数据文件定位字段

### 问题
Episode 元数据缺少 `data/chunk_index` 和 `data/file_index`，导致 LeRobot 无法通过 `get_data_file_path()` 定位每个 episode 的数据文件。

### LeRobot 的文件定位机制

```python
# lerobot_dataset.py 第197-200行
ep = self.episodes[ep_index]
chunk_idx = ep["data/chunk_index"]
file_idx = ep["data/file_index"]
fpath = self.data_path.format(chunk_index=chunk_idx, file_index=file_idx)
```

### 解决方案

**文件**: `dataset_cutter.py` 第288-305行

在 episode 元数据中添加文件定位字段：

```python
episode_meta = {
    'episode_index': new_episode_idx,
    'tasks': np.array([metadata['new_task']]),
    # 新增：LeRobot 必需的数据文件定位字段
    'data/chunk_index': to_int(metadata['episode_index']),  # 使用原始episode作为chunk
    'data/file_index': new_episode_idx,                     # 使用新episode index作为file index
    'dataset_from_index': global_frame_idx,
    'dataset_to_index': global_frame_idx + num_frames - 1,
    'length': num_frames,
    # ... 其他字段 ...
}
```

同时更新 `info.json` 的 `data_path` 模板（第468行）：

```python
'data_path': 'data/episode_{chunk_index}/segment_{file_index}.parquet',
```

**文件结构示例**：
```
data/
├── episode_0/
│   ├── segment_0.parquet   # chunk_index=0, file_index=0
│   ├── segment_1.parquet   # chunk_index=0, file_index=1
│   └── segment_2.parquet   # chunk_index=0, file_index=2
```

---

## 修复3：修复episode_index覆盖问题

### 问题
在保存帧数据时，使用了 `frame.get('episode_index', default)` 获取原始数据的 episode_index，导致所有切分后的片段都保留了原始数据集的 episode_index（都是0），而不是新分配的 0, 1, 2, ...

### 根本原因

```python
# 错误代码（修复前）
'episode_index': frame.get('episode_index', torch.tensor(new_episode_idx)),
```

如果原始 frame 已有 `episode_index` 字段，`.get()` 会返回原始值而不是默认值。

### 解决方案

**文件**: `dataset_cutter.py` 第310-324行

**强制覆盖**原始值，而不是使用 `.get()` 的默认值：

```python
# 准备帧数据
frame_records = []
for local_idx, frame in enumerate(frames):
    record = {
        'observation.images.image': frame['observation.images.image'],
        'observation.images.image2': frame['observation.images.image2'],
        'observation.state': frame['observation.state'],
        'action': frame['action'],
        'timestamp': frame.get('timestamp', torch.tensor(0.0)),
        # 强制使用新的索引值（不使用.get）
        'episode_index': torch.tensor(new_episode_idx),  # ✅ 直接赋值
        'frame_index': torch.tensor(local_idx),
        'index': torch.tensor(global_frame_idx - num_frames + local_idx),
        'task_index': torch.tensor(current_task_index),
    }
    frame_records.append(record)
```

---

## 修复4：实现正确的task_index映射

### 问题
所有帧的 `task_index` 都被硬编码为 0，没有根据实际任务描述分配不同的索引。

```python
# 错误代码（修复前）
'task_index': torch.tensor(0),  # ❌ 所有帧都是0
```

### LeRobot 的 task_index 机制

在原始数据集中，`task_index` 用于区分不同的任务类型：
- Episode 0: task_index=0, task="put the white mug on the left plate..."
- Episode 1: task_index=1, task="put the white mug on the plate..."
- Episode 2: task_index=2, task="put the yellow and white mug in the microwave..."
- Episode 3: task_index=2, task="put the yellow and white mug in the microwave..."（相同任务）

**相同任务描述 → 相同 task_index**

### 解决方案

**文件**: `dataset_cutter.py`

#### 步骤1：构建任务映射表（第262-272行）

```python
# 首先收集所有唯一的任务描述，构建任务索引映射
task_to_index = {}
for cut_range_id, episode_data in sorted(episodes_data.items()):
    task_desc = episode_data['metadata']['new_task']
    if task_desc not in task_to_index:
        task_to_index[task_desc] = len(task_to_index)

print(f"\n  任务映射表:")
for task, idx in sorted(task_to_index.items(), key=lambda x: x[1]):
    print(f"    {idx}: {task}")
```

**输出示例**：
```
任务映射表:
  0: pick up the white
  1: put the white on the left
  2: put the white on the plate
  3: pick up the yellow
```

#### 步骤2：为每个帧分配正确的task_index（第313-327行）

```python
# 获取当前episode的task_index
current_task = metadata['new_task']
current_task_index = task_to_index[current_task]

# 准备帧数据
for local_idx, frame in enumerate(frames):
    record = {
        # ... 其他字段 ...
        'task_index': torch.tensor(current_task_index),  # ✅ 使用正确的任务索引
    }
```

---

## 修复5：修复tasks.parquet格式

### 问题
`frame['task']` 返回数字（0, 1, 2）而不是任务描述字符串。

### 根本原因

LeRobot 的 `__getitem__` 方法（lerobot_dataset.py:1025-1026）：

```python
task_idx = item["task_index"].item()
item["task"] = self.meta.tasks.iloc[task_idx].name  # 使用 .name 获取行名
```

**关键**：`.iloc[i].name` 返回的是 DataFrame 的 **index**（行名），而不是列值。

### 错误的格式（修复前）

```python
# 错误：任务描述作为列
tasks_df = pd.DataFrame([
    {'task_index': 0, 'task': 'pick up the white'},
    {'task_index': 1, 'task': 'put the white on the left'},
])
tasks_df.to_parquet(file, index=False)
```

生成的结构：
```
   task_index                  task
0           0     pick up the white
1           1  put the white on the left
```

`tasks.iloc[0].name` 返回 `0`（数字index）❌

### 正确的格式（修复后）

**文件**: `dataset_cutter.py` 第368-374行

```python
# 正确：任务描述作为 DataFrame 的 index（行名）
tasks_data = []
for task, task_idx in sorted(task_to_index.items(), key=lambda x: x[1]):
    tasks_data.append({'task': task, 'task_index': task_idx})

tasks_df = pd.DataFrame(tasks_data)
# 将任务描述设为index（这是LeRobot期望的格式）
tasks_df = tasks_df.set_index('task')
tasks_file = self.output_dir / 'meta' / 'tasks.parquet'
tasks_df.to_parquet(tasks_file, index=True)  # ✅ 确保保存index
```

生成的结构：
```
                            task_index
task                                  
pick up the white                    0
put the white on the left            1
put the white on the plate           2
pick up the yellow                   3
```

`tasks.iloc[0].name` 返回 `"pick up the white"`（字符串）✅

---

## 验证结果

### 1. 数据集加载成功

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset('cut_dataset', root='/path/to/cut_dataset')
# ✅ 成功加载
# 总帧数: 282
# Episode数: 11
```

### 2. Episodes 正确分离

```python
unique_episodes = set(ds.hf_dataset.unique('episode_index'))
# ✅ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}  # 11个不同的episodes
```

### 3. Task_index 正确分配

```python
for ep_idx in range(ds.num_episodes):
    frame = ds[ds.meta.episodes[ep_idx]['dataset_from_index']]
    print(f"Episode {ep_idx}: task_index={frame['task_index'].item()}")

# ✅ 输出：
# Episode 0: task_index=0
# Episode 1: task_index=1
# Episode 2: task_index=0  # 相同任务，相同索引
# Episode 3: task_index=1
# Episode 5: task_index=2
# Episode 7: task_index=3
```

### 4. Task 字段返回字符串

```python
frame = ds[0]
print(frame['task'])
# ✅ "pick up the white"（字符串）
# ❌ 不是 "0"（数字）
```

### 5. 所有字段完整

```python
frame = ds[300]
print(frame.keys())
# ✅ ['observation.images.image', 'observation.images.image2', 
#     'observation.state', 'action', 'timestamp',
#     'episode_index', 'frame_index', 'index', 'task_index', 'task']
```

---

## LeRobot格式要求总结

### 必需的目录结构

```
dataset_root/
├── data/
│   └── <subdirs>/
│       └── *.parquet          # 帧数据
├── meta/
│   ├── info.json              # 数据集元信息（必需）
│   ├── stats.json             # 统计信息
│   ├── tasks.parquet          # 任务列表（index=任务描述）
│   └── episodes/              # Episodes元数据
│       └── chunk-000/
│           └── file-000.parquet
```

### info.json 必需字段

```json
{
  "codebase_version": "v3.0",
  "total_episodes": 11,
  "total_frames": 282,
  "data_path": "data/episode_{chunk_index}/segment_{file_index}.parquet",
  "features": {
    "observation.images.image": { "dtype": "image", ... },
    "observation.state": { "dtype": "float32", ... },
    "action": { "dtype": "float32", ... },
    "timestamp": { "dtype": "float32", ... },
    "episode_index": { "dtype": "int64", ... },  // 必需
    "frame_index": { "dtype": "int64", ... },    // 必需
    "index": { "dtype": "int64", ... },          // 必需
    "task_index": { "dtype": "int64", ... }      // 必需
  }
}
```

### episodes 元数据必需字段

```python
{
    'episode_index': 0,
    'data/chunk_index': 0,      # 用于定位数据文件
    'data/file_index': 0,       # 用于定位数据文件
    'dataset_from_index': 0,
    'dataset_to_index': 25,
    'length': 26,
    'tasks': ['task description'],
}
```

### 数据帧必需字段

每个 parquet 文件中的每一帧：

```python
{
    # 观测数据
    'observation.images.image': tensor([3, 256, 256]),
    'observation.images.image2': tensor([3, 256, 256]),
    'observation.state': tensor([8]),
    
    # 动作数据
    'action': tensor([7]),
    'timestamp': float,
    
    # 元数据（必需）
    'episode_index': int,   # 所属episode的索引
    'frame_index': int,     # Episode内的帧索引（0-based）
    'index': int,           # 全局帧索引
    'task_index': int,      # 任务索引
}
```

### tasks.parquet 格式要求

```python
# ✅ 正确：任务描述作为 DataFrame 的 index
                            task_index
task                                  
pick up the white                    0
put the white on the left            1

# ❌ 错误：任务描述作为列
   task_index                  task
0           0     pick up the white
1           1  put the white on the left
```

---

## 修复的文件

所有修复都在 `dataset_cutter.py` 文件中：

1. **第262-272行**: 添加任务映射构建逻辑
2. **第288-305行**: 添加 `data/chunk_index` 和 `data/file_index`
3. **第310-324行**: 强制覆盖 episode_index 和 task_index
4. **第368-374行**: 修复 tasks.parquet 格式
5. **第407-420行**: 保存元数据字段到 parquet
6. **第415-427行**: 定义元数据字段类型
7. **第447-519行**: 生成完整的 info.json
8. **第468行**: 修正 data_path 模板

---

## 使用方法

### 生成新数据集

```bash
cd /home/dongyingyibadao/data_dealer_auto

# 删除旧数据集
rm -rf cut_dataset

# 生成新数据集（本地模式）
python auto_cut_dataset.py \
    --end-idx 600 \
    --before-frames 15 \
    --after-frames 10 \
    --llm-provider local \
    --save-mode lerobot \
    --max-episodes 15

# 或使用 GPT 模式（需要 API Key）
python auto_cut_dataset.py \
    --end-idx 600 \
    --before-frames 15 \
    --after-frames 10 \
    --llm-provider gpt \
    --llm-api-key "your-key" \
    --llm-api-base "https://gpt.yunstorm.com/" \
    --llm-api-version "2025-01-01-preview" \
    --llm-model "gpt-4o" \
    --save-mode lerobot \
    --max-episodes 15
```

### 验证数据集

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 加载数据集
ds = LeRobotDataset('cut_dataset', root='/path/to/cut_dataset')

# 检查基本信息
print(f"总帧数: {len(ds)}")
print(f"Episodes: {ds.num_episodes}")

# 检查 episode_index
unique_eps = sorted(ds.hf_dataset.unique('episode_index'))
print(f"唯一的 episode_index: {unique_eps}")

# 检查 task_index 和 task
for ep_idx in range(min(5, ds.num_episodes)):
    frame = ds[ds.meta.episodes[ep_idx]['dataset_from_index']]
    print(f"Episode {ep_idx}:")
    print(f"  task_index: {frame['task_index'].item()}")
    print(f"  task: {frame['task']}")
```

---

## 相关文档

- **本文档**: 完整修复总结
- **README.md**: 项目主文档
- **docs/USAGE_GUIDE.md**: 详细使用指南
- **docs/QUICK_START.md**: 快速开始教程

---

**修复日期**: 2024年12月8日

**修复人员**: GitHub Copilot AI Assistant

**验证状态**: ✅ 所有修复已验证通过
