# 📁 data_dealer_auto - 项目结构

## 🎯 项目概述

`data_dealer_auto` 是一个自动化处理 Pick/Place 操作数据集的工具链。

**版本**: v1.1.0 (2025-12-14)

**核心功能**:
- 🔍 自动检测夹爪状态变化
- ✂️ 精准裁剪操作序列
- 🤖 AI生成任务描述
- 💾 LeRobot格式转换
- 🚀 官方API集成（3倍加速）
- 🔶 Placeholder支持

## 📂 目录结构

```
data_dealer_auto/
│
├── 📘 核心程序
│   ├── auto_cut_dataset.py                     # CLI主程序
│   ├── dataset_cutter.py                       # 数据裁剪和格式转换（官方API集成）
│   ├── task_description_generator.py           # 任务描述生成器
│   ├── gripper_detector.py                     # 夹爪状态检测
│   ├── lerobot_dataset_with_placeholder.py     # Placeholder运行时包装器
│   └── read_lerobot_dataset_simple.py          # 数据集验证工具
│
├── 📚 文档 (docs/)
│   ├── USAGE_GUIDE.md                          # 完整使用指南
│   ├── OFFICIAL_API_GUIDE.md                   # 官方API集成说明 ⭐ NEW
│   ├── LOAD_RANGES_GUIDE.md                    # --load-ranges详解
│   ├── CHECKPOINT_GUIDE.md                     # 断点续传指南
│   ├── LEROBOT_DATASET_PLACEHOLDER_USAGE.md    # Placeholder方案1
│   ├── PROMPT_CUSTOMIZATION_GUIDE.md           # LLM提示词定制
│   └── GITHUB_GUIDE.md                         # Git使用指南
│
├── 🛠️ 工具脚本 (scripts/)
│   ├── diagnose_gripper.py                     # 夹爪状态诊断
│   ├── diagnose_memory.py                      # 内存配置诊断
│   ├── visualize_merging.py                    # 可视化数据合并
│   └── run_with_checkpoint.sh                  # Checkpoint运行脚本
│
├── 🧪 测试 (tests/)
│   └── test_memory_optimization.py             # 内存优化测试
│
└── ⚙️ 配置文件
    ├── .gitignore                              # Git忽略规则
    ├── PROJECT_STRUCTURE.md                    # 本文档
    └── README.md                               # 项目主页
```

## 📖 核心文件说明

### 主程序

| 文件 | 功能 | 说明 |
|------|------|------|
| `auto_cut_dataset.py` | CLI接口 | 命令行参数解析、流程控制 |
| `dataset_cutter.py` | 核心引擎 | 数据裁剪、格式转换、官方API集成 |
| `task_description_generator.py` | AI描述生成 | 支持本地/Qwen/GPT-4o |
| `gripper_detector.py` | 关键帧检测 | 夹爪状态分析算法 |
| `lerobot_dataset_with_placeholder.py` | 运行时包装 | Placeholder方案1实现 |
| `read_lerobot_dataset_simple.py` | 验证工具 | 测试数据集加载 |

### 文档文件

| 文档 | 目标读者 | 内容 |
|------|---------|------|
| **USAGE_GUIDE.md** | 所有用户 | 基础使用、参数说明、案例 |
| **OFFICIAL_API_GUIDE.md** ⭐ | 所有用户 | 官方API使用、性能对比 |
| **LOAD_RANGES_GUIDE.md** | 中级用户 | 分离分析和转换、参数复用 |
| **CHECKPOINT_GUIDE.md** | 高级用户 | 断点续传、大规模数据处理 |
| **LEROBOT_DATASET_PLACEHOLDER_USAGE.md** | 高级用户 | 运行时Placeholder方案 |
| **PROMPT_CUSTOMIZATION_GUIDE.md** | 开发者 | 自定义LLM提示词 |
| **GITHUB_GUIDE.md** | 贡献者 | Git工作流程 |

## 🔄 数据流程

```
原始数据集
    ↓
[夹爪检测] gripper_detector.py
    ↓
frame_ranges_info.json
    ↓
[数据裁剪] dataset_cutter.py
    ├─ [方式1] 官方API（推荐） → HF_LEROBOT_HOME/
    └─ [方式2] 传统方法 → output_dir/
        ↓
LeRobot格式数据集
    ↓
[训练/测试] 可直接加载
```

## 📦 输出目录结构

### 使用官方API（推荐）

数据保存在 `$HF_LEROBOT_HOME/{repo_id}/`：

```
{repo_id}/
├── meta/
│   ├── info.json              # 数据集元信息
│   ├── tasks.parquet          # 任务列表
│   ├── stats.json             # 统计信息
│   └── episodes/              # Episode元数据
└── data/
    └── chunk-*/               # 数据文件
```

### 使用传统方法

数据保存在 `{output_dir}/`：

```
output_dir/
├── meta/
│   ├── info.json
│   ├── tasks.parquet
│   ├── stats.json
│   └── episodes/
│       └── chunk-000/
│           └── file-000.parquet
├── data/
│   └── episode_*/
│       └── segment_*.parquet
└── frame_ranges_info.json     # 分析结果
```

## 🚀 快速导航

### 我想...

- **开始使用** → [README.md](README.md) → [USAGE_GUIDE.md](docs/USAGE_GUIDE.md)
- **使用官方API** → [OFFICIAL_API_GUIDE.md](docs/OFFICIAL_API_GUIDE.md) ⭐
- **分离分析和转换** → [LOAD_RANGES_GUIDE.md](docs/LOAD_RANGES_GUIDE.md)
- **处理大规模数据** → [CHECKPOINT_GUIDE.md](docs/CHECKPOINT_GUIDE.md)
- **添加Placeholder** → [LEROBOT_DATASET_PLACEHOLDER_USAGE.md](docs/LEROBOT_DATASET_PLACEHOLDER_USAGE.md)
- **定制LLM提示词** → [PROMPT_CUSTOMIZATION_GUIDE.md](docs/PROMPT_CUSTOMIZATION_GUIDE.md)
- **贡献代码** → [GITHUB_GUIDE.md](docs/GITHUB_GUIDE.md)

## 🆕 版本历史

### v1.1.0 (2025-12-14)
- ✨ 集成 LeRobot 官方 API
- ⚡ 性能提升 3倍（10线程+5进程）
- 🔧 新增 `--repo-id` 参数
- 📝 新增 OFFICIAL_API_GUIDE.md

### v1.0.0
- 🎉 初始版本发布
- ✅ 基础数据裁剪功能
- ✅ Placeholder 支持
- ✅ Checkpoint 机制

## 📄 许可证

MIT License

---

**需要帮助？**
- 查看 [docs/](docs/) 目录
- 运行 `python auto_cut_dataset.py --help`
- 阅读 [USAGE_GUIDE.md](docs/USAGE_GUIDE.md)
