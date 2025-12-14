# 🤖 Data Dealer Auto

自动化处理 Pick/Place 操作数据集的工具链。

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LeRobot v3.0](https://img.shields.io/badge/LeRobot-v3.0-green.svg)](https://github.com/huggingface/lerobot)

## ✨ 核心功能

- 🔍 **智能检测** - 自动识别夹爪状态变化，定位 Pick/Place 关键帧
- ✂️ **精准裁剪** - 提取操作前后完整序列（可配置帧数）
- 🤖 **AI 任务描述** - 支持本地/Qwen/Azure GPT 生成任务描述  
- 💾 **LeRobot 格式** - 输出 LeRobot v3.0 标准格式，支持官方API
- 🔶 **占位符支持** - 标记同一 chunk 内的动作跳跃边界
- 🛡️ **断点保护** - Checkpoint 机制，支持中断恢复
- ⚡ **流式处理** - 内存优化，支持百万级帧数据
- 🚀 **性能优化** - 官方API并行加速（10线程+5进程）

## 📦 安装 


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

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dataset-path` | 输入数据集路径 | - |
| `--output-dir` | 输出目录 | `./cut_dataset` |
| `--batch-size` | 批处理大小 | 100 |
| `--before-frames` | 关键帧前的帧数 | 30 |
| `--after-frames` | 关键帧后的帧数 | 30 |
| `--llm-provider` | 任务描述生成 (`local`/`gpt`/`qwen`) | `local` |
| `--llm-fast-mode` | GPT快速模式（2帧图像） | False |
| `--save-mode` | 保存格式 (`lerobot`/`image`/`both`) | `lerobot` |
| `--repo-id` | HuggingFace repo ID | 自动生成 |
| `--insert-placeholders` | 物理插入placeholder | False |
| `--checkpoint-interval` | Checkpoint间隔 | 10 |

详细参数说明：`python auto_cut_dataset.py --help`

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
| `auto_cut_dataset.py` | 主程序：CLI接口和流程控制 |
| `dataset_cutter.py` | 核心算法：数据裁剪、格式转换、官方API集成 |
| `task_description_generator.py` | 任务描述生成器（支持GPT/Qwen） |
| `lerobot_dataset_with_placeholder.py` | 运行时Placeholder包装器（方案1） |
| `gripper_detector.py` | 夹爪状态检测算法 |
| `read_lerobot_dataset_simple.py` | 数据集验证工具 |

## 📁 项目结构

查看 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) 了解完整的项目结构。

## 📖 文档

查看 [`docs/`](docs/) 目录中的详细文档：

### 📘 使用指南
- **[USAGE_GUIDE.md](docs/USAGE_GUIDE.md)** - 完整使用指南和案例
- **[OFFICIAL_API_GUIDE.md](docs/OFFICIAL_API_GUIDE.md)** - LeRobot官方API集成说明 ⭐
- **[LOAD_RANGES_GUIDE.md](docs/LOAD_RANGES_GUIDE.md)** - --load-ranges参数详解

### 🔧 高级功能
- **[CHECKPOINT_GUIDE.md](docs/CHECKPOINT_GUIDE.md)** - 断点续传详解
- **[LEROBOT_DATASET_PLACEHOLDER_USAGE.md](docs/LEROBOT_DATASET_PLACEHOLDER_USAGE.md)** - Placeholder方案1（运行时）
- **[PROMPT_CUSTOMIZATION_GUIDE.md](docs/PROMPT_CUSTOMIZATION_GUIDE.md)** - LLM提示词定制

### 🚀 开发文档
- **[GITHUB_GUIDE.md](docs/GITHUB_GUIDE.md)** - Git使用指南

## 🐛 故障排除

### 内存不足
```bash
# 减小 batch_size
--batch-size 20
```

### 处理速度慢
```bash
# 增大 batch_size + 启用快速模式
--batch-size 100 --llm-fast-mode
```

### GPT API 问题
```bash
# 检查 API key、endpoint 和模型名
python test_azure_gpt.py
```

### 官方API初始化失败
程序会自动fallback到传统方法，或手动指定：
```bash
--use-traditional-method
```

详细故障排除：查看各个文档中的FAQ章节

## 🆕 更新日志

### v1.1.0 (2025-12-14)
- ✨ 集成LeRobot官方API（方案B）
- ⚡ 性能提升3倍（并行图片压缩）
- 🔧 添加`--repo-id`参数支持
- 📝 新增OFFICIAL_API_GUIDE.md文档

### v1.0.0
- 🎉 初始版本
- ✅ 基础数据裁剪功能
- ✅ Placeholder支持

## 📄 许可证

MIT License

---

**需要帮助？** 查看 [`docs/`](docs/) 目录或运行 `python auto_cut_dataset.py --help`
