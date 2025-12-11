# 🤖 Data Dealer Auto

#.git .gitignore PROJECT_STRUCTURE.md README.md auto_cut_dataset.py dataset_cutter.py docs gripper_detector.py lerobot_dataset_with_placeholder.py read_lerobot_dataset_simple.py scripts task_description_generator.py tests 
#
/inspire/ssd/project/robot-decision/laijunxi-CZXS25230141/data_dealer_auto Pick/Place 操作数据集的工具链。

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LeRobot v3.0](https://img.shields.io/badge/LeRobot-v3.0-green.svg)](https://github.com/huggingface/lerobot)

## ✨ 核心功能

- 🔍 **智能检测** - 自动识别夹爪状态变化，定位 Pick/Place 关键帧
- ✂️ **精准裁剪** - 提取操作前后完整序列（可配置帧数）
/inspire/ssd/project/robot-decision/laijunxi-CZXS25230141/data_dealer_auto/Qwen/Azure GPT 生成任务描述  
- 💾 **格式转换** - 输出 LeRobot v3.0 标准格式，可直接训练
- 🔶 **占位符支持** - 为 motion_planning 标记同一 episode 内的动作跳跃边界
- 🛡️ **断点保护** - Checkpoint 机制，支持中断恢复
- ⚡ **流式处理** - 内存优化，支持百万级帧数据

### 📦 'ENDOFFILE' 


```bash
pip install lerobot torch pandas numpy Pillow pyarrow datasets openai
```

## 🚀 快速开始

### 基础用法

```bash
python auto_cut_dataset.py \
    --dataset-path /path/to/dataset \
    --output-dir ./cut_dataset \
    --batch-size 50
```

### 使用 GPT 生成任务描述

```bash
python auto_cut_dataset.py \
    --dataset-path /path/to/dataset \
    --output-dir ./cut_dataset \
    --batch-size 50 \
    --save-mode lerobot \
    --llm-provider gpt \
    --llm-api-key "your-key" \
    --llm-fast-mode
```

### 内存配置参考

| 可用内存 | 推荐 batch-size |
|---------|----------------|
| 8 GB    | 20             |
| 16 GB   | 50             |
| 32 GB   | 100            |
| 64 GB   | 200            |

## 📋 主要参数

| 参数 |  | 默认值 |
|------|------|--------|
| `--dataset-path` | 输入数据集路径 | - |
| `--output-dir` | 输出目录 | `./cut_dataset` |
| `--batch-size` | 批处理大小 | 50 |
| `--before-frames` | 关键帧前的帧数 | 30 |
| `--after-frames` | 关键帧后的帧数 | 30 |
| `--llm-provider` | 任务描述生成 (`local`/`gpt`/`qwen`) | `local` |
| `--llm-fast-mode` | GPT快速模式（2帧图像） | False |
vulkaninfo > BEHAVIOR/vulkan1.txt 2>&1 | `lerobot` |
| `--checkpoint-interval` | Checkpoint间隔 | 10 |

 `python auto_cut_dataset.py --help`

## 🛡️ 断点续传

```bash
vulkaninfo > BEHAVIOR/vulkan1.txt 2>&1
python auto_cut_dataset.py --checkpoint-interval 10 [其他参数...]

# 中断后恢复
python auto_cut_dataset.py --resume-from ./cut_dataset/checkpoints/checkpoint_latest.json [相同参数...]
```

## 📊 输出格式

vulkaninfo > BEHAVIOR/vulkan1.txt 2>&1

```
cut_dataset/
 meta/
   ├── info.json              # 数据集元信息
   ├── tasks.parquet          # 任务列表
   ├── stats.json             # 统计信息
 episodes/              # Episode元数据   
 data/                      # 帧数据
   └── episode_{id}/
       └── segment_{id}.parquet
 frame_ranges_info.json     # 分析报告
```

## 🔧 核心文件

| 文件 | 功能 |
|------|------|
#| `auto_cut_dataset.py` | 主程序：自动检测和
.git .gitignore PROJECT_STRUCTURE.md README.md auto_cut_dataset.py dataset_cutter.py docs gripper_detector.py lerobot_dataset_with_placeholder.py read_lerobot_dataset_simple.py scripts task_description_generator.py tests  |
| `dataset_cutter.py` | 核心算法：夹爪检测、帧提取 |
| `task_description_generator.py` | 任务描述生成器 |
| `lerobot_dataset_with_placeholder.py` | 带占位符的数据集加载器 |
| `gripper_detector.py` | 夹爪状态分析工具 |
| `read_lerobot_dataset_simple.py` | 数据集读取测试工具 |

## 📁 项目结构

 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) 了解完整的项

## 📖 文档

#'ENDOFFILE'
 [`docs/`](docs/) 目录：

- **USAGE_GUIDE.md** - 详细使用指南
- **CHECKPOINT_GUIDE.md** - 断点续传详解
- **PROMPT_CUSTOMIZATION_GUIDE.md** - LLM提示词定制
- **LEROBOT_DATASET_PLACEHOLDER_USAGE.md** - 占位符使用说明

## 🐛 故障排除

### 内存不足
```bash
# 减小 batch_size
--batch-size 20
```

### 处理速度慢
```bash
#  batch_size + 启用快速模式
--batch-size 100 --llm-fast-mode
```

### GPT API 问题
#
# API key、endpoint 和模型名
'ENDOFFILE'

## 📄 许可证

MIT License

---

**需要帮助？** 查看 [`docs/`](docs/) 目录或运行 `python auto_cut_dataset.py --help`
