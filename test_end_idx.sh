#!/bin/bash
# 测试 --end-idx 参数

echo "测试1: 不指定 --end-idx（应该处理所有数据）"
echo "python auto_cut_dataset.py --help | grep -A 3 'end-idx'"
python auto_cut_dataset.py --help | grep -A 3 'end-idx'

echo ""
echo "==============================================="
echo "测试2: 显示默认行为"
echo "python auto_cut_dataset.py --skip-cutting"
echo "==============================================="
