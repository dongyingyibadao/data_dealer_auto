# 🤖 Data Dealer Auto

自动检测、裁剪和转换机器人 Pick/Place 操作数据集的完整工具链。

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LeRobot v3.0](https://img.shields.io/badge/LeRobot-v3.0-green.svg)](https://github.com/huggingface/lerobot)

## ✨ 核心功能

- 🔍 **智能检测** - 自动识别夹爪状态变化，定位 Pick/Place 关键帧
- ✂️ **精准裁剪** - 提取操作前后完整序列（可配置帧数）
- 🤖 **任务生成** - 支持本地规则/Qwen/Azure GPT 生成任务描述  
- 💾 **格式转换** - 输出 LeRobot v3.0 标准格式，可直接训练
- 🔶 **占位符支持** - 为 motion_planning 标记同一 episode 内的动作跳跃边界
- 🛡️ **断点保护** - Checkpoint 机制，支持中断恢复
- ⚡ **批量处理** - 高效处理大规模数据集

## 📦 快速安装

```bash
# 安装核心依赖
pip install lerobot torch pandas numpy Pillow pyarrow datasets

# （可选）LLM API 支持
pip install openai  # Azure GPT
```

## 🚀 5分钟上手

### 方式1：本地模式（无需API）

```bash
python auto_cut_dataset.py \
    --end-idx 600 \
    --max-episodes 15 \
    --save-mode lerobot
```

**输出**: `cut_dataset/` 目录包含完整的 LeRobot 格式数据

### 方式2：Azure GPT模式（推荐）

```bash
python auto_cut_dataset.py \
    --end-idx 600 \
    --max-episodes 15 \
    --llm-provider gpt \
    --llm-api-key "your-key" \
    --llm-api-base "https://gpt.yunstorm.com/" \
    --llm-api-version "2025-01-01-preview" \
    --llm-model "gpt-4o" \
    --save-mode lerobot
```

**优势**: 基于视觉理解生成精准任务描述

### 验证数据集

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 加载生成的数据集
dataset = LeRobotDataset(
    'cut_dataset',
    root='./cut_dataset'
)

print(f"总帧数: {len(dataset)}")
print(f"Episodes: {dataset.num_episodes}")

# 访问数据
frame = dataset[0]
print(f"Task: {frame['task']}")
print(f"Action: {frame['action']}")
```

### 使用占位符功能（用于 motion_planning）

```python
from lerobot_dataset_with_placeholder import LeRobotDatasetWithPlaceholder

# 加载带占位符的数据集
dataset = LeRobotDatasetWithPlaceholder(
    repo_id='cut_dataset',
    root='./cut_dataset',
    placeholder_action_value=-999.0
)

print(f"总帧数: {len(dataset)} (包含 {dataset.num_placeholders} 个占位符)")

# 遍历数据，占位符标记动作跳跃边界
for i in range(len(dataset)):
    frame = dataset[i]
    if frame['is_placeholder'].item():
        print(f"帧 {i}: 🔶 动作跳跃边界")
        # 重置轨迹缓冲区或其他逻辑
    else:
        # 正常处理观测和动作
        pass
```

**详细文档**: [LEROBOT_DATASET_PLACEHOLDER_USAGE.md](./docs/LEROBOT_DATASET_PLACEHOLDER_USAGE.md)

## 📊 输出格式

生成的数据集结构：

```
cut_dataset/
├── meta/
│   ├── info.json              # 数据集元信息
│   ├── tasks.parquet          # 任务列表
│   ├── stats.json             # 统计信息
│   └── episodes/              # Episode元数据
│       └── chunk-000/
│           └── file-000.parquet
├── data/                      # 帧数据
│   └── episode_{id}/
│       └── segment_{id}.parquet
└── frame_ranges_info.json    # 分析报告
```

## 🎯 核心参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--end-idx` | 处理的最大帧索引 | 所有帧 |
| `--max-episodes` | 最多保存的episodes | 所有 |
| `--before-frames` | 关键帧前的帧数 | 30 |
| `--after-frames` | 关键帧后的帧数 | 30 |
| `--llm-provider` | 任务描述生成方式 | `local` |
| `--save-mode` | 保存格式 | `lerobot` |
| `--checkpoint-interval` | Checkpoint间隔 | 10 |

完整参数列表: `python auto_cut_dataset.py --help`

## 🛡️ 断点续传

处理大数据集时使用 checkpoint 保护进度：

```bash
# 启用checkpoint
python auto_cut_dataset.py \
    --checkpoint-interval 10 \
    [其他参数...]

# 如果中断，从checkpoint恢复
python auto_cut_dataset.py \
    --resume-from ./cut_dataset/checkpoints/checkpoint_latest.json \
    [相同参数...]
```

或使用交互式脚本：

```bash
bash scripts/run_with_checkpoint.sh
```

## 📚 详细文档

- **[快速开始指南](docs/QUICK_START.md)** - 详细的入门教程
- **[使用手册](docs/USAGE_GUIDE.md)** - 完整参数说明和案例
- **[Checkpoint指南](docs/CHECKPOINT_GUIDE.md)** - 断点续传详解
- **[提示词定制](docs/PROMPT_CUSTOMIZATION_GUIDE.md)** - 自定义任务描述
- **[格式修复总结](FINAL_FIX_SUMMARY.md)** - LeRobot格式完整说明

## 🔧 核心组件

| 文件 | 功能 |
|------|------|
| `auto_cut_dataset.py` | 主程序入口 |
| `dataset_cutter.py` | 数据裁剪和格式转换 |
| `gripper_detector.py` | 夹爪状态检测 |
| `task_description_generator.py` | 任务描述生成 |
| `read_lerobot_dataset_simple.py` | 数据集读取工具 |

## ⚙️ 工作原理

### 1. Pick/Place 检测

监测动作向量第7维（夹爪状态）：
- **Pick**: -1.0 → 1.0 （夹爪关闭）
- **Place**: 1.0 → -1.0 （夹爪打开）

### 2. 帧范围提取

对每个关键帧 `i`，提取：
```
[i - before_frames, i + after_frames]
```

### 3. 任务描述生成

**本地模式**：规则生成
```
"pick the white mug"
"place the white mug on the left plate"
```

**GPT模式**：基于图像理解
- 上传6帧图像（2个摄像头 × 3个关键时刻）
- GPT-4o 视觉分析
- 生成准确的自然语言描述

### 4. 格式转换

输出符合 LeRobot v3.0 标准的 Parquet 数据：
- 必需字段：`episode_index`, `frame_index`, `index`, `task_index`
- 观测数据：`observation.images.image`, `observation.state`
- 动作数据：`action`, `timestamp`

## 🐛 故障排除

<details>
<summary><b>Q: 检测不到关键帧？</b></summary>

**可能原因**：
- 夹爪状态变化不明显
- 阈值设置不当

**解决方案**：
```python
# 编辑 gripper_detector.py，第29行
self.threshold = 0.3  # 降低阈值，更敏感
```
</details>

<details>
<summary><b>Q: GPT API 返回 401 错误？</b></summary>

**原因**：Vision 功能未启用或 API Key 无效

**解决方案**：
1. 验证 API Key 是否正确
2. 确认端点支持 Vision 功能
3. 或使用本地模式：`--llm-provider local`
</details>

<details>
<summary><b>Q: 磁盘空间不足？</b></summary>

**解决方案**：
```bash
# 限制处理范围
--end-idx 10000

# 限制episode数量
--max-episodes 100

# 使用外部存储
--output-dir /mnt/external/dataset
```
</details>

## 📊 性能参考

| 数据量 | Episodes | 处理时间 | 输出大小 |
|--------|----------|----------|----------|
| 600 帧 | 11 | ~2 分钟 | ~50 MB |
| 10k 帧 | ~180 | ~15 分钟 | ~800 MB |
| 100k 帧 | ~1800 | ~2.5 小时 | ~8 GB |
| 全量 (273k) | ~5000 | ~7 小时 | ~25 GB |

*基于本地模式，GPT模式约慢5-10倍*

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License

## 🔗 相关项目

- [LeRobot](https://github.com/huggingface/lerobot) - HuggingFace 机器人学习框架
- [LIBERO](https://libero-project.github.io/) - 机器人操作基准数据集

## 📚 文档索引

- [QUICK_START.md](./docs/QUICK_START.md) - 快速上手指南
- [USAGE_GUIDE.md](./docs/USAGE_GUIDE.md) - 详细使用文档
- [LEROBOT_DATASET_PLACEHOLDER_USAGE.md](./LEROBOT_DATASET_PLACEHOLDER_USAGE.md) - 占位符数据集使用指南
- [FINAL_FIX_SUMMARY.md](./FINAL_FIX_SUMMARY.md) - LeRobot 格式修复总结
- [PROMPT_CUSTOMIZATION_GUIDE.md](./PROMPT_CUSTOMIZATION_GUIDE.md) - LLM Prompt 定制指南

---

**需要帮助？** 查看 [详细文档](docs/) 或运行 `python auto_cut_dataset.py --help`
