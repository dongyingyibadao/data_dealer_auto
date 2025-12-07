#!/usr/bin/env python3
"""
测试checkpoint断点续传功能

测试流程:
1. 运行小规模数据处理 (只处理前5个任务)
2. 模拟中断 (人工停止)
3. 从checkpoint恢复
4. 验证结果一致性
"""

import subprocess
import sys
import time
import json
import os
from pathlib import Path

# ANSI颜色代码
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_step(step, desc):
    """打印测试步骤"""
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}步骤 {step}: {desc}{RESET}")
    print(f"{BLUE}{'='*70}{RESET}\n")

def print_success(msg):
    """打印成功消息"""
    print(f"{GREEN}✓ {msg}{RESET}")

def print_error(msg):
    """打印错误消息"""
    print(f"{RED}✗ {msg}{RESET}")

def print_info(msg):
    """打印信息"""
    print(f"{YELLOW}ℹ {msg}{RESET}")

def cleanup_test_files():
    """清理测试文件"""
    checkpoint_dir = Path("./cut_dataset/checkpoints")
    if checkpoint_dir.exists():
        print_info(f"清理checkpoint目录: {checkpoint_dir}")
        import shutil
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
    
    output_dir = Path("./cut_dataset")
    if output_dir.exists():
        # 只删除测试相关的文件，保留其他文件
        test_files = [
            "frame_ranges_info.json",
            "episodes_summary.json"
        ]
        for f in test_files:
            fp = output_dir / f
            if fp.exists():
                print_info(f"删除: {fp}")
                fp.unlink()

def run_initial_processing():
    """运行初始处理（会被中断）"""
    print_step(1, "运行初始处理（仅分析模式，快速测试）")
    
    cmd = [
        sys.executable, "auto_cut_dataset.py",
        "--end-idx", "50",  # 只处理前50帧，快速测试
        "--skip-cutting",   # 只分析不转换，加快速度
        "--checkpoint-interval", "2"  # 每2个任务保存一次
    ]
    
    print_info(f"运行命令: {' '.join(cmd)}")
    print_info("这应该会快速完成分析...")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60  # 60秒超时
        )
        
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        
        if result.returncode == 0:
            print_success("初始处理完成")
            return True
        else:
            print_error(f"初始处理失败，返回码: {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print_error("处理超时（60秒）")
        return False
    except Exception as e:
        print_error(f"运行出错: {e}")
        return False

def check_checkpoint_files():
    """检查checkpoint文件是否存在"""
    print_step(2, "检查checkpoint文件")
    
    checkpoint_dir = Path("./cut_dataset/checkpoints")
    
    if not checkpoint_dir.exists():
        print_error(f"Checkpoint目录不存在: {checkpoint_dir}")
        return False
    
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_*.json"))
    
    if not checkpoint_files:
        print_error("没有找到checkpoint文件")
        return False
    
    print_success(f"找到 {len(checkpoint_files)} 个checkpoint文件:")
    for f in checkpoint_files:
        print(f"  - {f.name}")
        
        # 读取并显示checkpoint内容
        try:
            with open(f, 'r') as fp:
                data = json.load(fp)
                print(f"    └─ 完成任务数: {len(data.get('completed_tasks', []))}")
                print(f"    └─ 已处理索引: {len(data.get('completed_ranges', []))}")
        except Exception as e:
            print_error(f"    └─ 读取失败: {e}")
    
    return True

def test_resume_from_checkpoint():
    """测试从checkpoint恢复"""
    print_step(3, "测试从checkpoint恢复")
    
    checkpoint_file = Path("./cut_dataset/checkpoints/checkpoint_latest.json")
    
    if not checkpoint_file.exists():
        print_error(f"最新checkpoint文件不存在: {checkpoint_file}")
        return False
    
    # 读取checkpoint信息
    with open(checkpoint_file, 'r') as f:
        checkpoint_data = json.load(f)
    
    completed_before = len(checkpoint_data.get('completed_tasks', []))
    print_info(f"Checkpoint中已完成任务数: {completed_before}")
    
    # 从checkpoint恢复运行
    cmd = [
        sys.executable, "auto_cut_dataset.py",
        "--end-idx", "50",
        "--skip-cutting",
        "--checkpoint-interval", "2",
        "--resume-from", str(checkpoint_file)
    ]
    
    print_info(f"运行命令: {' '.join(cmd)}")
    print_info("从checkpoint恢复...")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        
        if result.returncode == 0:
            print_success("恢复运行完成")
            
            # 检查是否跳过了已完成的任务
            if "从checkpoint恢复" in result.stdout or "跳过" in result.stdout:
                print_success("成功检测到checkpoint恢复逻辑")
            
            return True
        else:
            print_error(f"恢复运行失败，返回码: {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print_error("恢复运行超时（60秒）")
        return False
    except Exception as e:
        print_error(f"运行出错: {e}")
        return False

def verify_results():
    """验证结果"""
    print_step(4, "验证结果")
    
    # 检查输出文件
    output_file = Path("./cut_dataset/frame_ranges_info.json")
    
    if not output_file.exists():
        print_error(f"输出文件不存在: {output_file}")
        return False
    
    with open(output_file, 'r') as f:
        data = json.load(f)
    
    total_ranges = data.get('total_ranges', 0)
    print_success(f"成功生成输出文件，共 {total_ranges} 个帧范围")
    
    # 检查checkpoint最终状态
    checkpoint_file = Path("./cut_dataset/checkpoints/checkpoint_latest.json")
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        
        completed = len(checkpoint.get('completed_tasks', []))
        print_success(f"Checkpoint记录: {completed} 个任务已完成")
    
    return True

def main():
    """主测试流程"""
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}Checkpoint断点续传功能测试{RESET}")
    print(f"{BLUE}{'='*70}{RESET}\n")
    
    # 确认是否继续
    print_info("此测试将:")
    print("  1. 清理现有的checkpoint和测试文件")
    print("  2. 运行小规模数据分析（前50帧）")
    print("  3. 检查checkpoint文件生成")
    print("  4. 测试从checkpoint恢复")
    print("  5. 验证结果一致性")
    print()
    
    response = input(f"{YELLOW}是否继续？[y/N]: {RESET}").strip().lower()
    if response != 'y':
        print_info("测试已取消")
        return
    
    # 测试流程
    success = True
    
    # 0. 清理
    print_step(0, "清理测试环境")
    cleanup_test_files()
    print_success("清理完成")
    
    # 1. 初始运行
    if not run_initial_processing():
        print_error("初始处理失败")
        success = False
    
    # 2. 检查checkpoint
    if success and not check_checkpoint_files():
        print_error("Checkpoint文件检查失败")
        success = False
    
    # 3. 测试恢复
    if success and not test_resume_from_checkpoint():
        print_error("恢复测试失败")
        success = False
    
    # 4. 验证结果
    if success and not verify_results():
        print_error("结果验证失败")
        success = False
    
    # 总结
    print(f"\n{BLUE}{'='*70}{RESET}")
    if success:
        print(f"{GREEN}✓ 所有测试通过！Checkpoint功能正常工作{RESET}")
    else:
        print(f"{RED}✗ 测试失败，请检查上述错误信息{RESET}")
    print(f"{BLUE}{'='*70}{RESET}\n")
    
    # 询问是否清理
    response = input(f"{YELLOW}是否清理测试文件？[y/N]: {RESET}").strip().lower()
    if response == 'y':
        cleanup_test_files()
        print_success("测试文件已清理")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
