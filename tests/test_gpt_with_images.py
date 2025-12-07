"""
测试 GPT VLM 调用（带图像）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from task_description_generator import GPTVLM
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 配置
API_KEY = "5ffef770a5b148c5920b7b16329e30fa"
API_BASE = "https://gpt.yunstorm.com/"
API_VERSION = "2025-01-01-preview"

print("=" * 80)
print("测试 GPT VLM 调用（带图像）")
print("=" * 80)

# 加载数据集
print("\n📂 加载数据集...")
dataset = LeRobotDataset(
    repo_id="HuggingFaceVLA_cus/libero",
    root="/home/dongyingyibadao/HuggingFaceVLA_cus/libero"
)
print(f"✓ 数据集加载成功，共 {len(dataset)} 帧")

# 测试不同模型
models_to_test = [
    ("gpt-4o", False),      # 标准模式
    ("gpt-4o", True),       # 快速模式
    ("gpt-5", False),       # 标准模式
    ("gpt-5", True),        # 快速模式
]

for model_name, fast_mode in models_to_test:
    print("\n" + "=" * 80)
    print(f"测试模型: {model_name} {'(快速模式)' if fast_mode else '(标准模式)'}")
    print("=" * 80)
    
    # 创建 VLM
    vlm = GPTVLM(
        api_key=API_KEY,
        api_base=API_BASE,
        api_version=API_VERSION,
        model=model_name,
        fast_mode=fast_mode
    )
    
    # 准备测试数据
    context = {
        'episode_index': 0,
        'first_frame_cam1': dataset[100]['observation.images.image'],
        'last_frame_cam1': dataset[120]['observation.images.image'],
        'key_frame_cam1': dataset[110]['observation.images.image'],
        'first_frame_cam2': dataset[100]['observation.images.image2'],
        'last_frame_cam2': dataset[120]['observation.images.image2'],
        'key_frame_cam2': dataset[110]['observation.images.image2'],
    }
    
    # 测试生成
    try:
        print("🔄 调用 GPT VLM...")
        result = vlm.generate_task_description(
            action_type="pick",
            original_task="put the moka pot on the stove",
            context=context
        )
        print(f"✅ 成功!")
        print(f"   结果: {result}")
    except Exception as e:
        print(f"❌ 失败: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 80)
print("测试完成")
print("=" * 80)
