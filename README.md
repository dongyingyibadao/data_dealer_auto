# 📖 自动化Pick/Place数据集裁剪 - data_dealer_auto

这是一个完整的自动化Pick/Place操作检测和数据集转换工具，支持VLM任务描述生成和双模式数据保存。

## 🎯 快速开始

```bash
# 进入目录
cd /home/dongyingyibadao/data_dealer_auto

# 第一步：分析数据集（不进行裁剪，快速预览）
python auto_cut_dataset.py --end-idx 10000 --skip-cutting

# 第二步：查看分析结果
cat cut_dataset/frame_ranges_info.json

# 第三步：执行完整转换（使用GPT-4o VLM生成任务描述）
python auto_cut_dataset.py \
  --end-idx 10000 \
  --max-episodes 100 \
  --save-mode lerobot \
  --llm-provider gpt \
  --llm-api-key "your-api-key" \
  --llm-api-base "https://gpt.yunstorm.com/" \
  --llm-api-version "2025-01-01-preview" \
  --llm-model "gpt-4o"
```

## ✨ 核心功能

### ✅ 自动检测Pick/Place操作
- 监测夹爪状态变化（action[-1]）
- -1.0 → 1.0：Pick操作（夹爪关闭）
- 1.0 → -1.0：Place操作（夹爪打开）

### ✅ 灵活的帧范围提取
- 可自定义关键帧前后帧数
- `--before-frames`: 关键帧前取的帧数（默认30）
- `--after-frames`: 关键帧后取的帧数（默认30）
- 自动处理边界情况和episode切换

### ✅ VLM智能任务描述生成
- **GPT-4o视觉理解**：上传6张图片（两个摄像头×3帧）
  - Camera 1（整体场景）+ Camera 2（操作细节）
  - 首帧、关键帧、尾帧
- **智能物体识别**：准确识别操作对象和位置
- **支持复合形容词**：如"yellow and white mug"（黄白相间的杯子）
- **备选方案**：Qwen、Deepseek或本地规则生成

### ✅ 双模式数据保存
1. **图片模式** (`--save-mode image`)
   - 保存为JPEG图片，方便人工检查
   - 两个摄像头的所有帧
   - JSON元数据

2. **LeRobot Parquet模式** (`--save-mode lerobot`)
   - 完全符合LeRobot标准格式
   - 图像编码为PNG bytes
   - 可直接用于训练

3. **双模式** (`--save-mode both`)
   - 同时保存两种格式

## 📁 文件结构

```
data_dealer_auto/
├── auto_cut_dataset.py              # 主程序入口
├── gripper_detector.py              # 夹爪检测模块
├── task_description_generator.py    # VLM任务描述生成
├── dataset_cutter.py                # 数据裁剪和保存
├── diagnose_gripper.py              # 夹爪诊断工具
├── visualize_merging.py             # 可视化工具
├── QUICK_START.md                   # 快速开始指南
├── README.md                        # 本文件：完整说明
├── USAGE_GUIDE.md                   # 详细使用指南
├── INDEX.md                         # 项目索引
├── PROMPT_CUSTOMIZATION_GUIDE.md    # Prompt自定义指南
└── cut_dataset/                     # 输出目录（首次运行后创建）
    ├── frame_ranges_info.json       # 帧范围分析结果
    ├── images/                      # 图片模式输出
    │   ├── episode_0000/
    │   │   ├── frame_0000_cam1.jpg
    │   │   ├── frame_0000_cam2.jpg
    │   │   └── ...
    │   └── episodes_summary.json
    ├── meta/                        # LeRobot模式元数据
    │   ├── episodes/chunk-000/
    │   │   ├── file-000.parquet
    │   │   ├── info.json
    │   │   └── stats.json
    │   └── tasks.parquet
    └── data/                        # LeRobot模式数据
        └── episode_0/
            ├── segment_0.parquet
            ├── segment_1.parquet
            └── ...
```

## 📊 工作流程

```
输入数据集 (LeRobot格式)
         ↓
    [步骤1] 检测夹爪状态变化
         ↓
    [步骤2] 提取指定帧范围
         ↓
    [步骤3] VLM生成任务描述
         ↓
    [步骤4] 保存为指定格式
         ↓
输出数据集 (图片/LeRobot格式)
```

## ⚡ 性能参考

- 检测速度：~1,000 帧/秒
- VLM任务生成：~5-10 秒/episode（取决于网络）
- 数据转换：~500 帧/秒
- 内存占用：1-4 GB

处理10,000帧（约15个episode）：约2-5分钟

## 📋 主要参数

### 基本参数
```bash
--dataset-path PATH          # 输入数据集路径
--output-dir PATH            # 输出目录（默认：./cut_dataset）
--start-idx N                # 开始帧索引（默认：0）
--end-idx N                  # 结束帧索引（默认：10000）
--max-episodes N             # 最多保存的episode数量
```

### 帧提取参数
```bash
--before-frames N            # 关键帧前取的帧数（默认：30）
--after-frames N             # 关键帧后取的帧数（默认：30）
```

### 保存模式
```bash
--save-mode {image,lerobot,both}
  image: 保存为JPEG图片（方便检查）
  lerobot: 保存为Parquet格式（用于训练）
  both: 同时保存两种格式
```

### VLM任务描述生成
```bash
--llm-provider {gpt,qwen,deepseek,local}  # LLM提供者
--llm-api-key KEY                         # API密钥
--llm-api-base URL                        # API基础URL
--llm-api-version VERSION                 # API版本（Azure OpenAI）
--llm-model MODEL                         # 模型名称（如：gpt-4o）
```

### 其他参数
```bash
--skip-cutting               # 仅分析，不转换数据
--load-ranges FILE           # 加载之前保存的分析结果
```

## 🧪 示例用法
  ✓ 完整工作流可以成功执行
  ✅ 所有测试通过！

📖 详细文档
─────────────────────────────────────────────────────────────────────────────

README.md              - 完整技术文档和参数说明
USAGE_GUIDE.md         - 详细使用案例和故障排除
QUICK_START.md         - 快速启动指南

🎓 使用示例
─────────────────────────────────────────────────────────────────────────────

示例1：快速分析（仅分析，不转换）
  python auto_cut_dataset.py --end-idx 10000 --skip-cutting

示例2：完整处理（分析+转换）
  python auto_cut_dataset.py --end-idx 10000 --max-episodes 100

示例3：处理更多数据
  python auto_cut_dataset.py --end-idx 100000 --max-episodes 500

示例4：使用Qwen生成更好的任务描述
  python auto_cut_dataset.py \
### 示例1：快速测试（仅分析，不保存数据）
```bash
python auto_cut_dataset.py --end-idx 1000 --skip-cutting
```

### 示例2：保存为图片格式（方便检查）
```bash
python auto_cut_dataset.py \
  --end-idx 5000 \
  --max-episodes 10 \
  --save-mode image
```

### 示例3：使用GPT-4o VLM生成任务描述并保存为LeRobot格式
```bash
python auto_cut_dataset.py \
  --llm-provider gpt \
  --llm-api-key "your-api-key" \
  --llm-api-base "https://gpt.yunstorm.com/" \
  --llm-api-version "2025-01-01-preview" \
  --llm-model "gpt-4o" \
  --before-frames 40 \
  --after-frames 20 \
  --end-idx 10000 \
  --max-episodes 100 \
  --save-mode lerobot
```

### 示例4：同时保存两种格式
```bash
python auto_cut_dataset.py \
  --llm-provider gpt \
  --llm-api-key "your-api-key" \
  --llm-api-base "https://gpt.yunstorm.com/" \
  --llm-api-version "2025-01-01-preview" \
  --llm-model "gpt-4o" \
  --end-idx 10000 \
  --save-mode both
```

### 示例5：重复使用之前的分析结果
```bash
python auto_cut_dataset.py \
  --load-ranges cut_dataset/frame_ranges_info.json \
  --max-episodes 200 \
  --save-mode lerobot
```

## 💡 最佳实践

### 1. 分步骤执行
- ① 先用 `--skip-cutting` 分析
- ② 查看 `frame_ranges_info.json` 确认结果
- ③ 确认无误后再执行数据保存

### 2. 处理大数据集
- ① 先处理1,000-5,000帧测试
- ② 确认参数和输出后扩大范围
- ③ 使用 `--max-episodes` 限制输出大小

### 3. VLM使用建议
- ① 先用小数据集测试API连接
- ② 检查生成的任务描述质量
- ③ 根据需要调整Prompt（见PROMPT_CUSTOMIZATION_GUIDE.md）

### 4. 保存模式选择
- **开发/调试阶段**：使用 `image` 模式方便检查
- **训练准备阶段**：使用 `lerobot` 模式
- **需要两者**：使用 `both` 模式

## 🔍 验证输出

### 检查图片模式输出
```bash
ls -lh cut_dataset/images/episode_0000/
cat cut_dataset/episodes_summary.json
```

### 检查LeRobot模式输出
```bash
python -c "
import pandas as pd
df = pd.read_parquet('cut_dataset/meta/episodes/chunk-000/file-000.parquet')
print(df)
"
```

### 用LeRobot加载数据集
```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    repo_id="local",
    root="./cut_dataset"
)
print(f"Total frames: {len(dataset)}")
print(f"Sample: {dataset[0]}")
```

## 📞 获取帮助

查看完整参数列表：
```bash
python auto_cut_dataset.py --help
```

查看文档：
- `README.md` - 本文件
- `QUICK_START.md` - 快速入门
- `USAGE_GUIDE.md` - 详细使用指南  
- `PROMPT_CUSTOMIZATION_GUIDE.md` - VLM Prompt自定义
- `INDEX.md` - 项目索引

## 🐛 常见问题

### Q: VLM生成的任务描述不准确？
**A**: 检查以下几点：
1. 确认使用的是 `gpt-4o` 模型（支持视觉）
2. 查看 `PROMPT_CUSTOMIZATION_GUIDE.md` 优化Prompt
3. 检查图像是否正确提取（可先用 `image` 模式查看）

### Q: 为什么检测到的episode数量少？
**A**: 可能原因：
1. `--end-idx` 设置太小
2. 数据集中Pick/Place操作较少
3. 调整 `--before-frames` 和 `--after-frames` 参数

### Q: LeRobot格式能否被原版LeRobot加载？
**A**: 可以！输出格式完全兼容LeRobot标准：
```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
dataset = LeRobotDataset(repo_id="local", root="./cut_dataset")
```

### Q: 如何只重新生成任务描述？
**A**: 使用 `--load-ranges` 参数：
```bash
python auto_cut_dataset.py \
  --load-ranges cut_dataset/frame_ranges_info.json \
  --llm-provider gpt \
  --llm-api-key "your-key"
```

## 📊 输出文件说明

### frame_ranges_info.json
包含所有检测到的Pick/Place操作信息：
```json
{
  "total_ranges": 10,
  "frame_ranges": [
    {
      "id": 0,
      "keyframe_index": 40,
      "action_type": "pick",
      "frame_start": 10,
      "frame_end": 71,
      "num_frames": 61,
      "original_task": "put the mug on the plate",
      "new_task": "pick the white mug",
      "episode_index": 0,
      "frame_index": 40
    }
  ],
  "pick_count": 5,
  "place_count": 5
}
```

## 🎓 相关文档

- [QUICK_START.md](QUICK_START.md) - 5分钟快速入门
- [USAGE_GUIDE.md](USAGE_GUIDE.md) - 详细使用说明
- [PROMPT_CUSTOMIZATION_GUIDE.md](PROMPT_CUSTOMIZATION_GUIDE.md) - VLM Prompt优化
- [INDEX.md](INDEX.md) - 完整项目索引

---

**项目路径**: `/home/dongyingyibadao/data_dealer_auto`  
**版本**: 2.0  
**最后更新**: 2025-12-03


rm -rf /inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA_cus/datasets_cut_image && time python auto_cut_dataset.py --dataset-path /inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA_cus/libero --output-dir /inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA_cus/datasets_cut_image --end-idx 30000 --before-frames 15 --after-frames 10 --llm-provider gpt --llm-api-key 5ffef770a5b148c5920b7b16329e30fa --llm-api-base https://gpt.yunstorm.com/ --llm-api-version 2025-01-01-preview --save-mode image --llm-model gpt-4o 2>&1 | tee data_cut_image.log