#!/bin/bash
# 断点续传功能 - 使用示例脚本

echo "=========================================="
echo "🔄 断点续传功能演示"
echo "=========================================="
echo

# 配置
OUTPUT_DIR="/inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA_cus/datasets_cut"
API_KEY="5ffef770a5b148c5920b7b16329e30fa"
API_BASE="https://gpt.yunstorm.com/"
API_VERSION="2025-01-01-preview"
MODEL="gpt-4o"

# 检查点目录
CHECKPOINT_DIR="$OUTPUT_DIR/checkpoints"
LATEST_CHECKPOINT="$CHECKPOINT_DIR/checkpoint_latest.json"

echo "📊 配置信息："
echo "   输出目录: $OUTPUT_DIR"
echo "   检查点目录: $CHECKPOINT_DIR"
echo "   API模型: $MODEL"
echo

# 检查是否存在检查点
if [ -f "$LATEST_CHECKPOINT" ]; then
    echo "✓ 发现检查点文件: $LATEST_CHECKPOINT"
    echo
    
    # 显示检查点信息
    echo "📖 检查点信息："
    cat "$LATEST_CHECKPOINT" | python3 -m json.tool | grep -E '(timestamp|progress|completed_count|error)'
    echo
    
    # 询问是否从检查点恢复
    read -p "是否从检查点恢复？(y/n): " RESUME
    echo
    
    if [ "$RESUME" = "y" ] || [ "$RESUME" = "Y" ]; then
        echo "▶️  从检查点恢复运行..."
        echo
        
        time python auto_cut_dataset.py \
          --llm-provider gpt \
          --llm-api-key "$API_KEY" \
          --llm-api-base "$API_BASE" \
          --llm-api-version "$API_VERSION" \
          --llm-model "$MODEL" \
          --checkpoint-interval 10 \
          --output-dir "$OUTPUT_DIR" \
          --resume-from "$LATEST_CHECKPOINT"
    else
        echo "⏩ 跳过恢复，从头开始运行"
        echo
        
        read -p "确认要从头开始吗？这将忽略现有检查点 (y/n): " CONFIRM
        echo
        
        if [ "$CONFIRM" = "y" ] || [ "$CONFIRM" = "Y" ]; then
            echo "🗑️  清理旧检查点..."
            rm -rf "$CHECKPOINT_DIR"
            echo "✓ 旧检查点已清理"
            echo
            
            echo "▶️  开始全新运行..."
            echo
            
            time python auto_cut_dataset.py \
              --llm-provider gpt \
              --llm-api-key "$API_KEY" \
              --llm-api-base "$API_BASE" \
              --llm-api-version "$API_VERSION" \
              --llm-model "$MODEL" \
              --checkpoint-interval 10 \
              --output-dir "$OUTPUT_DIR"
        else
            echo "❌ 取消运行"
            exit 0
        fi
    fi
else
    echo "ℹ️  未发现检查点文件，开始全新运行"
    echo
    
    echo "▶️  开始处理..."
    echo
    
    time python auto_cut_dataset.py \
      --llm-provider gpt \
      --llm-api-key "$API_KEY" \
      --llm-api-base "$API_BASE" \
      --llm-api-version "$API_VERSION" \
      --llm-model "$MODEL" \
      --checkpoint-interval 10 \
      --output-dir "$OUTPUT_DIR"
fi

echo
echo "=========================================="
echo "✅ 完成！"
echo "=========================================="
echo
echo "📁 输出位置: $OUTPUT_DIR"
echo "📁 检查点位置: $CHECKPOINT_DIR"
echo

# 显示最终统计
if [ -f "$OUTPUT_DIR/frame_ranges_info.json" ]; then
    echo "📊 结果统计："
    cat "$OUTPUT_DIR/frame_ranges_info.json" | python3 -m json.tool | grep -E '(total_ranges|pick_count|place_count)'
fi
