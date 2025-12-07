# ⚡ 快速开始 - data_dealer_auto

5分钟上手自动化Pick/Place数据集处理工具

## 📦 前置要求

- Python 3.8+
- LeRobot库已安装
- 数据集路径：`/home/dongyingyibadao/HuggingFaceVLA_cus/libero`

## 🚀 三步快速开始

### 第一步：分析数据集（不保存数据）

```bash
cd /home/dongyingyibadao/data_dealer_auto

python auto_cut_dataset.py \
  --end-idx 1000 \
  --skip-cutting
```

**输出**: `cut_dataset/frame_ranges_info.json` - 包含所有检测到的Pick/Place操作

### 第二步：查看分析结果

```bash
cat cut_dataset/frame_ranges_info.json | python -m json.tool | head -30
```

你会看到类似输出：
```json
{
  "total_ranges": 5,
  "frame_ranges": [
    {
      "id": 0,
      "keyframe_index": 40,
      "action_type": "pick",
      "frame_start": 10,
      "frame_end": 71,
      "num_frames": 61,
      "original_task": "put the mug on the plate",
      "new_task": "pick object",
      "episode_index": 0
    }
  ]
}
```

### 第三步：保存数据

#### 选项A：保存为图片（方便检查）

```bash
python auto_cut_dataset.py \
  --end-idx 1000 \
  --max-episodes 5 \
  --save-mode image
```

查看结果：
```bash
ls -lh cut_dataset/images/episode_0000/
```

#### 选项B：保存为LeRobot格式（用于训练）

```bash
python auto_cut_dataset.py \
  --end-idx 1000 \
  --max-episodes 5 \
  --save-mode lerobot
```

验证结果：
```bash
python -c "
import pandas as pd
df = pd.read_parquet('cut_dataset/meta/episodes/chunk-000/file-000.parquet')
print(df)
"
```

## 🎯 使用VLM生成任务描述

如果你有GPT-4o API访问权限：

```bash
python auto_cut_dataset.py \
  --llm-provider gpt \
  --llm-api-key "your-api-key" \
  --llm-api-base "https://gpt.yunstorm.com/" \
  --llm-api-version "2025-01-01-preview" \
  --llm-model "gpt-4o" \
  --end-idx 1000 \
  --max-episodes 5 \
  --save-mode lerobot
```

**VLM会分析6张图片**（两个摄像头×3帧）来准确识别操作对象。

## 📊 常用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--end-idx N` | 处理的帧数 | 10000 |
| `--max-episodes N` | 最多保存的episode数 | 无限制 |
| `--before-frames N` | 关键帧前取的帧数 | 30 |
| `--after-frames N` | 关键帧后取的帧数 | 30 |
| `--save-mode` | `image`/`lerobot`/`both` | lerobot |
| `--skip-cutting` | 仅分析不保存 | False |

## 💡 小贴士

### 1. 先小规模测试
```bash
# 先处理1000帧看看效果
python auto_cut_dataset.py --end-idx 1000 --skip-cutting
```

### 2. 调整帧范围
```bash
# 如果动作较快，可以减少帧数
python auto_cut_dataset.py --before-frames 20 --after-frames 15
```

### 3. 重用分析结果
```bash
# 第一次：分析
python auto_cut_dataset.py --end-idx 5000 --skip-cutting

# 第二次：使用之前的分析结果，只保存数据
python auto_cut_dataset.py \
  --load-ranges cut_dataset/frame_ranges_info.json \
  --save-mode image
```

## ✅ 验证输出

### 检查图片模式
```bash
# 查看生成的图片
ls cut_dataset/images/episode_0000/

# 查看元数据
cat cut_dataset/episodes_summary.json | python -m json.tool
```

### 检查LeRobot模式
```bash
# 查看episodes元数据
python -c "
import pandas as pd
df = pd.read_parquet('cut_dataset/meta/episodes/chunk-000/file-000.parquet')
print('Episodes:')
print(df[['episode_index', 'action_type', 'length']])
"

# 查看tasks
python -c "
import pandas as pd
df = pd.read_parquet('cut_dataset/meta/tasks.parquet')
print('Tasks:')
print(df)
"
```

## 📚 下一步

- 查看 [README.md](README.md) 了解完整功能
- 查看 [USAGE_GUIDE.md](USAGE_GUIDE.md) 了解详细用法
- 查看 [PROMPT_CUSTOMIZATION_GUIDE.md](PROMPT_CUSTOMIZATION_GUIDE.md) 优化VLM

---

**项目路径**: `/home/dongyingyibadao/data_dealer_auto`
