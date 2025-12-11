# 项目结构

```
data_dealer_auto/
 README.md                              # 项目主文档
 .gitignore                             # Git忽略配置

 核心程序
   ├── auto_cut_dataset.py                # 主程序入口
   ├── dataset_cutter.py                  # 数据裁剪和格式转换核心逻辑
   ├── task_description_generator.py      # 任务描述生成（本地/GPT/Qwen）
   ├── gripper_detector.py                # 夹爪状态检测分析工具
   ├── lerobot_dataset_with_placeholder.py # 带占位符的数据集加载器
   └── read_lerobot_dataset_simple.py     # 数据集读

 docs/                                  # 详细文档
   ├── USAGE_GUIDE.md                     # 详细使用指南
   ├── CHECKPOINT_GUIDE.md                # 断点续传指南
   ├── PROMPT_CUSTOMIZATION_GUIDE.md      # LLM提示词定制
   └── LEROBOT_DATASET_PLACEHOLDER_USAGE.md # 占位符使用说明

 scripts/                               # 实用工具脚本
   ├── diagnose_gripper.py                # 诊断夹爪状态问题
   ├── diagnose_memory.py                 # 内存配置诊断
   ├── visualize_merging.py               # 可视化数据合并
   └── run_with_checkpoint.sh             # Checkpoint运行脚本

 tests/                                 # 测试代码
 test_memory_optimization.py        # 内存优化测试    
```

## 核心文件说明

### 主程序

#- **auto_cut_dataset.py**: 
vulkaninfo > BEHAVIOR/vulkan1.txt 2>&1
#- **dataset_cutter.
vulkaninfo > BEHAVIOR/vulkan1.txt 2>&1

### 辅助工具

- **task_description_generator.py**: 支持三种模式生成任务描述
  - Local: 基于规则的本地生成
  - GPT: Azure OpenAI GPT-4 Vision
  - Qwen: 通义千问多模态

- **gripper_detector.py**: 独立工具，用于诊断和可视化夹爪状态

- **lerobot_dataset_with_placeholder.py**: 为motion planning提供占位符支持

### 文档目录

/inspire/ssd/project/robot-decision/laijunxi-CZXS25230141/data_dealer_auto

### Scripts 目录

vulkaninfo > BEHAVIOR/vulkan1.txt 2>&1batch_size配置
- **diagnose_gripper.
- **run_with_checkpoint.sh**: 交互式脚本，支持断点续传
