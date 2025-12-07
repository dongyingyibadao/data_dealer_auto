#!/usr/bin/env python3
"""
诊断脚本：检查夹爪状态变化的详细情况
"""

from pathlib import Path
import pandas as pd
import numpy as np

def diagnose_gripper_states(dataset_path, max_frames=2000):
    """诊断夹爪状态"""
    
    print("=" * 80)
    print("🔬 夹爪状态诊断 (直接读取Parquet)")
    print("=" * 80)
    
    # 直接读取Parquet文件
    data_dir = Path(dataset_path) / "data" / "chunk-000"
    parquet_files = sorted(data_dir.glob("*.parquet"))
    
    if not parquet_files:
        print(f"❌ 未找到Parquet文件在: {data_dir}")
        return
    
    print(f"📂 读取Parquet文件: {len(parquet_files)} 个")
    print(f"📊 采样上限: {max_frames} 帧\n")
    
    # 读取所有帧（直到达到上限）
    all_frames = []
    total_rows = 0
    
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        print(f"  ✓ {pf.name}: {len(df)} 行")
        all_frames.append(df)
        total_rows += len(df)
        
        if total_rows >= max_frames:
            break
    
    df_all = pd.concat(all_frames, ignore_index=True)
    if len(df_all) > max_frames:
        df_all = df_all.iloc[:max_frames]
    
    print(f"\n✓ 总加载帧数: {len(df_all)}\n")
    
    # 提取夹爪状态
    print("=" * 80)
    print("📈 分析结果")
    print("=" * 80)
    
    print(f"\n1️⃣  Parquet文件结构:")
    print(f"   列名: {list(df_all.columns)}")
    
    # 查找action列
    action_cols = [c for c in df_all.columns if 'action' in c.lower()]
    print(f"\n   Action相关列: {action_cols}")
    
    # 提取夹爪状态
    if 'action' in df_all.columns:
        actions = df_all['action'].values
        gripper_states = []
        
        for action in actions:
            if isinstance(action, np.ndarray):
                if len(action) >= 7:
                    gripper_states.append(float(action[-1]))
                else:
                    gripper_states.append(None)
            elif isinstance(action, (list, tuple)):
                if len(action) >= 7:
                    gripper_states.append(float(action[-1]))
                else:
                    gripper_states.append(None)
            else:
                gripper_states.append(None)
        
        gripper_states = [g for g in gripper_states if g is not None]
        
        print(f"\n2️⃣  采样数据统计:")
        print(f"   - 总采样数: {len(gripper_states)}")
        print(f"   - 夹爪状态范围: {min(gripper_states):.4f} ~ {max(gripper_states):.4f}")
        print(f"   - 夹爪状态均值: {np.mean(gripper_states):.4f}")
        print(f"   - 夹爪状态中位数: {np.median(gripper_states):.4f}")
        
        # 统计夹爪状态分布
        open_count = sum(1 for g in gripper_states if g < -0.5)
        close_count = sum(1 for g in gripper_states if g > 0.5)
        middle_count = sum(1 for g in gripper_states if -0.5 <= g <= 0.5)
        
        print(f"\n   - 夹爪打开状态(<-0.5): {open_count} ({100*open_count/len(gripper_states):.1f}%)")
        print(f"   - 夹爪关闭状态(>0.5): {close_count} ({100*close_count/len(gripper_states):.1f}%)")
        print(f"   - 夹爪中间状态(-0.5~0.5): {middle_count} ({100*middle_count/len(gripper_states):.1f}%)")
        
        # 检测状态变化
        print(f"\n3️⃣  夹爪状态变化统计:")
        changes = []
        threshold = 0.5
        
        for i in range(1, len(gripper_states)):
            prev = gripper_states[i-1]
            curr = gripper_states[i]
            diff = abs(curr - prev)
            
            if diff > threshold:
                action_type = 'unknown'
                if prev < 0 and curr > 0:
                    action_type = 'pick'
                elif prev > 0 and curr < 0:
                    action_type = 'place'
                
                changes.append({
                    'index': i,
                    'prev': prev,
                    'curr': curr,
                    'diff': diff,
                    'type': action_type
                })
        
        print(f"   - 总变化数 (threshold={threshold}): {len(changes)}")
        
        pick_count = sum(1 for c in changes if c['type'] == 'pick')
        place_count = sum(1 for c in changes if c['type'] == 'place')
        unknown_count = sum(1 for c in changes if c['type'] == 'unknown')
        
        print(f"   - Pick变化 (打开→关闭): {pick_count}")
        print(f"   - Place变化 (关闭→打开): {place_count}")
        print(f"   - 未知变化: {unknown_count}")
        
        if pick_count + place_count > 0:
            ratio = place_count / (pick_count + place_count)
            print(f"   - Place/(Pick+Place): {ratio:.2%}")
        
        print(f"\n4️⃣  前15个变化详情:")
        for i, change in enumerate(changes[:15]):
            print(f"   [{i:2d}] 索引{change['index']:4d}: " +
                  f"{change['prev']:7.4f} → {change['curr']:7.4f} " +
                  f"({change['diff']:6.4f}) = {change['type']}")
        
        if changes:
            print(f"\n5️⃣  变化差值分布:")
            diffs = [c['diff'] for c in changes]
            print(f"   - 最小差值: {min(diffs):.4f}")
            print(f"   - 最大差值: {max(diffs):.4f}")
            print(f"   - 平均差值: {np.mean(diffs):.4f}")
            print(f"   - 中位数差值: {np.median(diffs):.4f}")
        
        # 查看task信息
        if 'task' in df_all.columns:
            print(f"\n6️⃣  Task信息:")
            tasks = df_all['task'].unique()
            print(f"   - 不同任务数: {len(tasks)}")
            print(f"   - 前5个任务:")
            for task in tasks[:5]:
                count = (df_all['task'] == task).sum()
                print(f"     - '{task[:40]}' ({count} 帧)")
    
    else:
        print(f"\n❌ 未找到'action'列")
        print(f"   可用列: {list(df_all.columns)}")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    dataset_path = '/home/dongyingyibadao/HuggingFaceVLA_cus/libero'
    diagnose_gripper_states(dataset_path, max_frames=2000)
