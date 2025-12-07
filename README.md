# 🤖 data_dealer_auto - 自动化Pick/Place数据处理工具

一个完整的自动化系统，用于检测机器人操作中的Pick/Place关键帧，生成任务描述，并转换为LeRobot标准格式。

---

## 🚀 快速开始

### 基础用法

```bash
# 1. 快速分析（不裁剪，仅检测）
python auto_cut_dataset.py --end-idx 10000 --skip-cutting

# 2. 完整处理（带任务描述生成）
python auto_cut_dataset.py \
  --llm-provider gpt \
  --llm-api-key "your-api-key" \
  --llm-api-base "https://gpt.yunstorm.com/" \
  --llm-api-version "2025-01-01-preview" \
  --llm-model "gpt-4o"

# 3. 使用checkpoint功能（防止数据丢失）
python auto_cut_dataset.py [参数] --checkpoint-interval 10
# 如果中断，恢复运行：
python auto_cut_dataset.py [参数] --resume-from ./cut_dataset/checkpoints/checkpoint_latest.json
```

### 带交互式恢复的运行脚本

```bash
# 使用便捷脚本，自动检测并询问是否恢复
bash scripts/run_with_checkpoint.sh
```

---

## ✨ 核心功能

### 🎯 自动检测Pick/Place操作
- 监测夹爪状态变化（`action[-1]`）
- **Pick操作**：-1.0 → 1.0（夹爪关闭）
- **Place操作**：1.0 → -1.0（夹爪打开）
- 可自定义关键帧前后帧数（默认各30帧）

### 🧠 VLM智能任务描述生成
- **GPT-4o视觉理解**：上传6张图片分析
  - 双摄像头 × 3关键帧（首帧、关键帧、尾帧）
  - 准确识别操作对象和位置
  - 支持复合形容词（如"yellow and white mug"）
- **备选方案**：Qwen、Deepseek或本地规则生成

### 💾 Checkpoint恢复机制
- 自动保存进度（默认每10个任务）
- 错误时立即保存checkpoint
- 支持断点续传，避免长时间运行时数据丢失
- 详细的恢复日志

### 📦 双模式数据保存
1. **图片模式**（`--save-mode image`）：JPEG格式，便于检查
2. **LeRobot模式**（`--save-mode lerobot`）：Parquet格式，可直接训练
3. **双模式**（`--save-mode both`）：同时保存两种格式

---

## 📁 项目结构

```
data_dealer_auto/
├── README.md                    # 本文件
├── auto_cut_dataset.py          # 主程序入口
├── gripper_detector.py          # 夹爪检测模块
├── task_description_generator.py # VLM任务描述生成
├── dataset_cutter.py            # 数据裁剪和保存
│
├── docs/                        # 📚 所有文档
│   ├── QUICK_START.md          # 快速开始指南
│   ├── USAGE_GUIDE.md          # 详细使用指南
│   ├── CHECKPOINT_GUIDE.md     # Checkpoint功能完整指南
│   ├── CHECKPOINT_QUICK_REF.txt # Checkpoint快速参考
│   ├── CHECKPOINT_IMPLEMENTATION.md # Checkpoint实现说明
│   ├── GPT_FAST_MODE_GUIDE.md  # GPT快速模式指南
│   ├── FAST_MODE_QUICK_REF.txt # 快速模式参考
│   ├── PROMPT_CUSTOMIZATION_GUIDE.md # Prompt自定义指南
│   ├── GITHUB_GUIDE.md         # GitHub使用指南
│   ├── GITHUB_QUICK_START.md   # GitHub快速参考
│   └── CHANGELOG_END_IDX.md    # end-idx参数更新日志
│
├── scripts/                     # 🔧 辅助脚本
│   ├── run_with_checkpoint.sh  # 交互式checkpoint恢复脚本
│   ├── visualize_merging.py    # 数据可视化工具
│   └── diagnose_gripper.py     # 夹爪诊断工具
│
├── tests/                       # 🧪 测试脚本
│   ├── test_azure_gpt.py       # Azure GPT API测试
│   ├── test_fast_mode.py       # 快速模式测试
│   ├── test_gpt_with_images.py # GPT图像处理测试
│   ├── test_minimal_vlm.py     # VLM最小测试
│   ├── test_text_vs_image.py   # 文本vs图像测试
│   └── test_end_idx.sh         # end-idx参数测试
│
└── cut_dataset/                 # 📂 输出目录（运行后自动创建）
    ├── frame_ranges_info.json  # 帧范围分析结果
    ├── checkpoints/            # Checkpoint文件
    ├── images/                 # 图片模式输出
    ├── meta/                   # LeRobot元数据
    └── data/                   # LeRobot数据
```

---

## 📖 文档导航

| 文档 | 用途 | 适用场景 |
|------|------|----------|
| **README.md** | 项目概览和快速入门 | 首次使用 |
| [QUICK_START.md](docs/QUICK_START.md) | 3步快速启动 | 想立即开始 |
| [USAGE_GUIDE.md](docs/USAGE_GUIDE.md) | 详细参数说明和案例 | 需要深入了解 |
| [CHECKPOINT_GUIDE.md](docs/CHECKPOINT_GUIDE.md) | Checkpoint完整指南 | 处理大数据集 |
| [CHECKPOINT_QUICK_REF.txt](docs/CHECKPOINT_QUICK_REF.txt) | Checkpoint快速参考卡 | 快速查询 |
| [GPT_FAST_MODE_GUIDE.md](docs/GPT_FAST_MODE_GUIDE.md) | GPT快速模式详解 | 优化性能 |
| [PROMPT_CUSTOMIZATION_GUIDE.md](docs/PROMPT_CUSTOMIZATION_GUIDE.md) | 自定义Prompt | 定制任务描述 |
| [GITHUB_GUIDE.md](docs/GITHUB_GUIDE.md) | Git/GitHub操作 | 版本控制 |

---

## 🎓 常见使用场景

### 场景1: 快速测试（100条数据）
```bash
python auto_cut_dataset.py --end-idx 100 --skip-cutting
```

### 场景2: 完整处理 + GPT-4o描述
```bash
python auto_cut_dataset.py \
  --llm-provider gpt \
  --llm-api-key "5ffef770a5b148c5920b7b16329e30fa" \
  --llm-api-base "https://gpt.yunstorm.com/" \
  --llm-api-version "2025-01-01-preview" \
  --llm-model "gpt-4o"
```

### 场景3: 大数据集 + Checkpoint保护（270k帧）
```bash
python auto_cut_dataset.py \
  --llm-provider gpt \
  --llm-api-key "your-key" \
  --llm-api-base "https://gpt.yunstorm.com/" \
  --llm-model "gpt-4o" \
  --checkpoint-interval 10
```

### 场景4: 恢复中断的任务
```bash
python auto_cut_dataset.py [原参数] \
  --resume-from ./cut_dataset/checkpoints/checkpoint_latest.json
```

---

## 🔧 主要参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--end-idx` | 处理的最大帧索引 | None（全部） |
| `--max-episodes` | 最大episode数量 | 100 |
| `--save-mode` | 保存模式 | lerobot |
| `--skip-cutting` | 仅分析不转换 | False |
| `--llm-provider` | LLM提供商 | local |
| `--llm-model` | 模型名称 | gpt-4o |
| `--checkpoint-interval` | checkpoint间隔 | 10 |
| `--resume-from` | checkpoint文件路径 | None |

完整参数列表请参见 [USAGE_GUIDE.md](docs/USAGE_GUIDE.md)

---

## 🛠️ 辅助工具

### 🔍 夹爪诊断工具
```bash
python scripts/diagnose_gripper.py
```
用于检查夹爪状态变化和统计信息。

### 📊 数据可视化
```bash
python scripts/visualize_merging.py
```
可视化帧范围合并过程。

### 🔄 交互式Checkpoint恢复
```bash
bash scripts/run_with_checkpoint.sh
```
自动检测checkpoint文件并询问是否恢复。

---

## 🧪 测试

所有测试脚本位于 `tests/` 目录：

```bash
# 测试Azure GPT API
python tests/test_azure_gpt.py

# 测试快速模式
python tests/test_fast_mode.py

# 测试图像处理
python tests/test_gpt_with_images.py
```

---

## 🐛 故障排除

### Q: 运行中断后如何恢复？
A: 使用 `--resume-from ./cut_dataset/checkpoints/checkpoint_latest.json`

### Q: 如何查看checkpoint状态？
A: 查看 `./cut_dataset/checkpoints/checkpoint_latest.json` 文件

### Q: Vision API返回401错误？
A: 当前endpoint不支持Vision功能，使用text-only模式或切换endpoint

详细问题解答请参见 [USAGE_GUIDE.md](docs/USAGE_GUIDE.md#常见问题)

---

## 📝 License

本项目遵循 MIT License。

## 🤝 贡献

欢迎提交Issue和Pull Request！

GitHub仓库：https://github.com/dongyingyibadao/data_dealer_auto

---

## 📮 联系方式

如有问题，请在GitHub提交Issue或联系项目维护者。

---

**最后更新**: 2025-12-07
