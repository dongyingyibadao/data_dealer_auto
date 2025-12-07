#!/usr/bin/env python3
"""
可视化脚本：展示帧范围合并的过程和影响
"""

import json
from pathlib import Path

def visualize_merging():
    """可视化合并过程"""
    
    print("=" * 80)
    print("📊 帧范围合并过程可视化")
    print("=" * 80)
    
    # 读取生成的frame_ranges_info.json
    info_file = Path("/home/dongyingyibadao/data_dealer_auto/cut_dataset/frame_ranges_info.json")
    
    if not info_file.exists():
        print(f"❌ 文件不存在：{info_file}")
        print("请先运行：python auto_cut_dataset.py --end-idx 10000 --skip-cutting")
        return
    
    with open(info_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    ranges = data['frame_ranges']
    
    print(f"\n✓ 加载了 {len(ranges)} 个合并后的范围\n")
    
    # 分析每个范围
    print("📋 前15个范围的详细信息：\n")
    print(f"{'ID':<3} {'Type':<6} {'Key':<6} {'Start':<6} {'End':<6} "
          f"{'Frames':<7} {'Task':<30}")
    print("-" * 90)
    
    for i, r in enumerate(ranges[:15]):
        task = r['original_task'][:27].ljust(27)
        print(f"{i:<3} {r['action_type']:<6} {r['keyframe_index']:<6} "
              f"{r['frame_start']:<6} {r['frame_end']:<6} {r['num_frames']:<7} {task}")
    
    print("\n" + "=" * 80)
    print("📈 统计信息")
    print("=" * 80)
    
    # 统计
    pick_ranges = [r for r in ranges if r['action_type'] == 'pick']
    place_ranges = [r for r in ranges if r['action_type'] == 'place']
    
    print(f"\n总范围数：{len(ranges)}")
    print(f"  • Pick操作：{len(pick_ranges)} ({len(pick_ranges)*100/len(ranges):.1f}%)")
    print(f"  • Place操作：{len(place_ranges)} ({len(place_ranges)*100/len(ranges):.1f}%)")
    
    # 帧数统计
    total_frames = sum(r['num_frames'] for r in ranges)
    avg_frames = total_frames / len(ranges)
    
    print(f"\n帧数统计：")
    print(f"  • 总帧数：{total_frames}")
    print(f"  • 平均每个范围：{avg_frames:.1f} 帧")
    print(f"  • 最小：{min(r['num_frames'] for r in ranges)} 帧")
    print(f"  • 最大：{max(r['num_frames'] for r in ranges)} 帧")
    
    # 范围重叠分析
    print(f"\n🔍 范围重叠分析：")
    
    overlaps = 0
    consecutive_picks = 0
    consecutive_places = 0
    
    for i in range(1, len(ranges)):
        prev_r = ranges[i-1]
        curr_r = ranges[i]
        
        # 检查重叠
        if prev_r['frame_end'] > curr_r['frame_start']:
            overlaps += 1
        
        # 检查相邻同类型操作
        if (prev_r['action_type'] == curr_r['action_type'] and 
            curr_r['frame_start'] - prev_r['frame_end'] < 50):
            if curr_r['action_type'] == 'pick':
                consecutive_picks += 1
            else:
                consecutive_places += 1
    
    print(f"  • 存在重叠的范围对：{overlaps}")
    if overlaps > 0:
        print(f"    这表明：同一个操作的多个状态变化被捕捉")
    
    print(f"\n📌 关键发现：")
    print(f"  • 原始检测：138 个关键帧")
    print(f"  • 合并后：{len(ranges)} 个范围")
    print(f"  • 合并比例：{(138-len(ranges))/138*100:.1f}%")
    print(f"  • 这表明：有 {138-len(ranges)} 个操作被合并到了附近的操作中")
    
    # 任务类型分析
    print(f"\n🎯 任务分布：")
    tasks = {}
    for r in ranges:
        task = r['original_task']
        if task not in tasks:
            tasks[task] = {'pick': 0, 'place': 0}
        tasks[task][r['action_type']] += 1
    
    for task, counts in sorted(tasks.items())[:3]:
        print(f"  • '{task[:40]}...'")
        print(f"    - Pick: {counts['pick']}, Place: {counts['place']}")
    
    print("\n" + "=" * 80)
    print("💡 关于合并的思考")
    print("=" * 80)
    
    print("""
为什么要进行合并？

1. 减少冗余数据
   • 原始138个关键帧检测到了许多短时间内的Pick-Place对
   • 这些操作在物理上非常接近（<50帧）
   • 合并避免了重复存储相同的帧数据

2. 逻辑完整性
   • 机器人任务 = Pick某物 + Place到某地
   • 检测到的多个夹爪状态变化描述的是同一个操作
   • 合并后形成逻辑上完整的操作单元

3. 节省存储空间
   • 138个范围需要存储每个帧数据的副本
   • 合并成39个后，避免了大量重复的Parquet块

4. 改进任务描述
   • 分离的范围：多个不完整的任务描述
   • 合并的范围：一个完整的"抓起并放下"描述
    """)
    
    print("=" * 80)


if __name__ == '__main__':
    visualize_merging()
