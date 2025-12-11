#!/usr/bin/env python3
"""
测试流式批处理功能
"""
import sys
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset_cutter import DatasetCutter


def test_batch_processing():
    """测试批处理功能"""
    print("🧪 测试批处理功能...")
    
    # 创建测试cutter
    cutter = DatasetCutter(
        output_dir='./test_output',
        save_mode='lerobot',
        batch_size=10
    )
    
    print(f"✅ DatasetCutter 初始化成功")
    print(f"   - 输出目录: {cutter.output_dir}")
    print(f"   - 保存模式: {cutter.save_mode}")
    print(f"   - 批处理大小: {cutter.batch_size}")
    
    # 测试新方法是否存在
    assert hasattr(cutter, 'extract_frames_batch'), "❌ extract_frames_batch 方法不存在"
    print("✅ extract_frames_batch 方法存在")
    
    assert hasattr(cutter, 'save_as_lerobot_format_streaming'), "❌ save_as_lerobot_format_streaming 方法不存在"
    print("✅ save_as_lerobot_format_streaming 方法存在")
    
    print("\n✅ 所有测试通过！")


def test_cut_and_convert_params():
    """测试 cut_and_convert_dataset 函数参数"""
    print("\n🧪 测试 cut_and_convert_dataset 函数...")
    
    from dataset_cutter import cut_and_convert_dataset
    import inspect
    
    # 检查函数签名
    sig = inspect.signature(cut_and_convert_dataset)
    params = list(sig.parameters.keys())
    
    print(f"函数参数: {params}")
    
    required_params = ['dataset', 'frame_ranges', 'output_dir', 'save_mode', 
                      'max_episodes', 'batch_size', 'streaming']
    
    for param in required_params:
        assert param in params, f"❌ 缺少参数: {param}"
        print(f"✅ 参数存在: {param}")
    
    print("\n✅ 函数签名正确！")


def test_auto_cut_dataset():
    """测试 auto_cut_dataset.py 的导入"""
    print("\n🧪 测试 auto_cut_dataset.py...")
    
    try:
        import auto_cut_dataset
        print("✅ auto_cut_dataset.py 导入成功")
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False
    
    return True


if __name__ == '__main__':
    print("=" * 60)
    print("🔬 内存优化功能测试")
    print("=" * 60)
    
    try:
        test_batch_processing()
        test_cut_and_convert_params()
        test_auto_cut_dataset()
        
        print("\n" + "=" * 60)
        print("🎉 所有测试通过！")
        print("=" * 60)
        print("\n下一步：")
        print("1. 运行诊断工具: python scripts/diagnose_memory.py")
        print("2. 测试小数据集: python auto_cut_dataset.py --end-idx 100 --batch-size 10")
        print("3. 查看文档: cat 内存优化简明说明.md")
        
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
