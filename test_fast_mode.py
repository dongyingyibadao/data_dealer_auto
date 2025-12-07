"""
测试GPT快速模式 vs 精细模式
"""
import time
from task_description_generator import TaskDescriptionGenerator

# 测试数据
test_range = {
    'action_type': 'pick',
    'task': 'put both moka pots on the stove',
    'episode_index': 376,
    'keyframe_index': 100,
    'frame_start': 70,
    'frame_end': 130
}

# Azure OpenAI 配置
API_KEY = "5ffef770a5b148c5920b7b16329e30fa"
API_BASE = "https://gpt.yunstorm.com/"
API_VERSION = "2025-01-01-preview"
MODEL = "gpt-5"

print("=" * 60)
print("GPT 快速模式 vs 精细模式测试")
print("=" * 60)

# 测试精细模式
print("\n🔍 测试精细模式（6帧图像）...")
generator_fine = TaskDescriptionGenerator(
    provider='gpt',
    api_key=API_KEY,
    api_base=API_BASE,
    api_version=API_VERSION,
    model=MODEL,
    fast_mode=False
)
print(f"  配置: fast_mode={generator_fine.llm.fast_mode}")

# 测试快速模式
print("\n⚡ 测试快速模式（2帧图像）...")
generator_fast = TaskDescriptionGenerator(
    provider='gpt',
    api_key=API_KEY,
    api_base=API_BASE,
    api_version=API_VERSION,
    model=MODEL,
    fast_mode=True
)
print(f"  配置: fast_mode={generator_fast.llm.fast_mode}")

print("\n" + "=" * 60)
print("✅ 配置测试完成")
print("=" * 60)
print("\n说明：")
print("  - 精细模式：上传6帧（cam1和cam2各3帧）")
print("  - 快速模式：上传2帧（cam1首尾帧）")
print("  - 预期速度提升：约3倍")
print("  - 预期API成本降低：约66%")
print("\n要实际测试，需要加载数据集并调用 generate_descriptions()")
