# 🔄 断点续传功能使用指南

## 📖 概述

为了防止长时间运行过程中出现错误导致所有工作白费，我们添加了**断点续传功能**：

✅ **自动保存检查点** - 每处理10个任务自动保存进度  
✅ **错误自动保存** - 出现错误时立即保存当前进度  
✅ **从断点恢复** - 使用 `--resume-from` 参数从断点继续  
✅ **灵活的保存间隔** - 使用 `--checkpoint-interval` 自定义保存频率  

---

## 🚀 快速开始

### 1. 正常运行（自动保存检查点）

```bash
python auto_cut_dataset.py \
  --llm-provider gpt \
  --llm-api-key YOUR_KEY \
  --llm-api-base https://gpt.yunstorm.com/ \
  --llm-api-version 2025-01-01-preview \
  --llm-model gpt-4o
```

检查点会自动保存在：`./cut_dataset/checkpoints/`

### 2. 从检查点恢复

如果运行中断，使用以下命令继续：

```bash
python auto_cut_dataset.py \
  --llm-provider gpt \
  --llm-api-key YOUR_KEY \
  --llm-api-base https://gpt.yunstorm.com/ \
  --llm-api-version 2025-01-01-preview \
  --llm-model gpt-4o \
  --resume-from ./cut_dataset/checkpoints/checkpoint_latest.json
```

---

## 📁 检查点文件说明

### 检查点保存位置

```
cut_dataset/
└── checkpoints/
    ├── checkpoint_latest.json              ← 最新检查点（用于恢复）
    ├── checkpoint_progress_20251207_143052_idx19.json
    ├── checkpoint_progress_20251207_143122_idx29.json
    ├── checkpoint_error_20251207_143210_idx430.json    ← 错误时保存
    └── checkpoint_final.json               ← 完成时保存
```

### 检查点文件内容

```json
{
  "timestamp": "20251207_143210",
  "last_index": 430,
  "total": 4038,
  "progress": "431/4038",
  "completed_count": 431,
  "completed_ranges": [...],
  "error": false
}
```

---

## ⚙️ 参数说明

### `--checkpoint-interval`
**说明**：每处理多少个任务保存一次检查点  
**默认值**：10  
**用法**：
```bash
# 每5个保存一次（更频繁，更安全但略慢）
--checkpoint-interval 5

# 每50个保存一次（较快，但失败时损失更多）
--checkpoint-interval 50
```

### `--resume-from`
**说明**：从指定的检查点文件恢复  
**用法**：
```bash
# 使用最新检查点
--resume-from ./cut_dataset/checkpoints/checkpoint_latest.json

# 使用特定检查点
--resume-from ./cut_dataset/checkpoints/checkpoint_error_20251207_143210_idx430.json
```

---

## 📊 使用场景

### 场景 1：处理大量数据（27万帧）

```bash
# 启动处理
python auto_cut_dataset.py \
  --llm-provider gpt \
  --llm-api-key YOUR_KEY \
  --llm-api-base https://gpt.yunstorm.com/ \
  --llm-api-version 2025-01-01-preview \
  --llm-model gpt-4o \
  --checkpoint-interval 10

# 如果中途报错或中断，从检查点恢复
python auto_cut_dataset.py \
  --llm-provider gpt \
  --llm-api-key YOUR_KEY \
  --llm-api-base https://gpt.yunstorm.com/ \
  --llm-api-version 2025-01-01-preview \
  --llm-model gpt-4o \
  --resume-from ./cut_dataset/checkpoints/checkpoint_latest.json
```

### 场景 2：不稳定的网络环境

```bash
# 更频繁地保存检查点（每5个）
python auto_cut_dataset.py \
  --llm-provider gpt \
  --llm-api-key YOUR_KEY \
  --llm-model gpt-4o \
  --checkpoint-interval 5
```

### 场景 3：测试后继续

```bash
# 先测试一小部分
python auto_cut_dataset.py \
  --end-idx 100 \
  --llm-provider gpt \
  --llm-model gpt-4o

# 确认没问题后，处理全部数据
python auto_cut_dataset.py \
  --llm-provider gpt \
  --llm-model gpt-4o \
  --resume-from ./cut_dataset/checkpoints/checkpoint_latest.json
```

---

## 🔍 监控进度

### 查看检查点信息

```bash
# 查看最新检查点
cat ./cut_dataset/checkpoints/checkpoint_latest.json | grep "progress"

# 列出所有检查点
ls -lh ./cut_dataset/checkpoints/
```

### 实时监控（在另一个终端）

```bash
# 监控检查点目录
watch -n 5 'ls -lht ./cut_dataset/checkpoints/ | head -10'

# 监控进度
watch -n 5 'tail -1 ./cut_dataset/checkpoints/checkpoint_latest.json'
```

---

## ⚠️ 注意事项

### 1. 检查点只保存任务描述生成进度

- ✅ 保存：任务描述生成的进度
- ❌ 不保存：夹爪检测、数据集裁剪的进度

如果在其他步骤出错，需要从头开始。

### 2. 恢复时参数必须一致

确保恢复时使用相同的参数：
- `--llm-provider`
- `--llm-api-key`
- `--llm-model`
- `--llm-fast-mode`

### 3. 检查点文件较大

每个检查点包含所有已完成的结果，文件会随着进度增大：
- 1000个任务 ≈ 2-5 MB
- 4000个任务 ≈ 8-20 MB

确保有足够的磁盘空间。

### 4. 自动清理旧检查点

建议定期清理不需要的检查点：

```bash
# 删除除最新和最终检查点外的所有检查点
cd ./cut_dataset/checkpoints
ls | grep -v 'checkpoint_latest\|checkpoint_final' | xargs rm
```

---

## 🐛 故障排除

### Q: 恢复时提示"无法找到检查点文件"

**解决**：检查文件路径是否正确
```bash
ls -l ./cut_dataset/checkpoints/checkpoint_latest.json
```

### Q: 恢复后从头开始而不是从断点

**解决**：确保使用了 `--resume-from` 参数
```bash
python auto_cut_dataset.py ... --resume-from ./cut_dataset/checkpoints/checkpoint_latest.json
```

### Q: 恢复时报错"参数不匹配"

**解决**：确保恢复时使用的参数与原始运行一致

### Q: 检查点文件损坏

**解决**：使用之前的检查点
```bash
# 查看所有检查点
ls -lt ./cut_dataset/checkpoints/

# 使用较早的检查点
--resume-from ./cut_dataset/checkpoints/checkpoint_progress_20251207_142000_idx400.json
```

---

## 📈 性能建议

### 检查点保存间隔选择

| 处理速度 | 建议间隔 | 说明 |
|---------|---------|------|
| 快速（local模式） | 50-100 | 每个任务很快，不需要频繁保存 |
| 中等（API无图像） | 20-50 | 平衡保存频率和性能 |
| 慢速（GPT VLM） | 5-10 | API调用慢，频繁保存更安全 |

### 磁盘空间规划

- 检查点目录：预留 100 MB
- 最终输出：根据数据量预留（参考 USAGE_GUIDE.md）

---

## ✅ 完整示例

### 处理全部27万帧数据（带断点续传）

```bash
#!/bin/bash
# 第一次运行
time python auto_cut_dataset.py \
  --llm-provider gpt \
  --llm-api-key 5ffef770a5b148c5920b7b16329e30fa \
  --llm-api-base https://gpt.yunstorm.com/ \
  --llm-api-version 2025-01-01-preview \
  --llm-model gpt-4o \
  --checkpoint-interval 10 \
  --output-dir /inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA_cus/datasets_cut

# 如果中断，使用以下命令恢复
# time python auto_cut_dataset.py \
#   --llm-provider gpt \
#   --llm-api-key 5ffef770a5b148c5920b7b16329e30fa \
#   --llm-api-base https://gpt.yunstorm.com/ \
#   --llm-api-version 2025-01-01-preview \
#   --llm-model gpt-4o \
#   --checkpoint-interval 10 \
#   --output-dir /inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA_cus/datasets_cut \
#   --resume-from /inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA_cus/datasets_cut/checkpoints/checkpoint_latest.json
```

---

## 📞 需要帮助？

遇到问题？检查：
1. 检查点文件是否存在
2. 参数是否一致
3. 磁盘空间是否充足
4. 查看完整错误信息

详细文档：`USAGE_GUIDE.md` 和 `README.md`
