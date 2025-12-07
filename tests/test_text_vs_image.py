"""
测试：对比文本调用和图像调用
"""
from openai import AzureOpenAI
import base64
import io
from PIL import Image

# 配置
API_KEY = "5ffef770a5b148c5920b7b16329e30fa"
API_BASE = "https://gpt.yunstorm.com/"
API_VERSION = "2025-01-01-preview"

print("=" * 80)
print("测试：对比文本调用和图像调用")
print("=" * 80)

# 创建客户端
client = AzureOpenAI(
    api_key=API_KEY,
    azure_endpoint=API_BASE,
    api_version=API_VERSION
)

# 创建测试图像
img = Image.new('RGB', (256, 256), color=(73, 109, 137))
buffered = io.BytesIO()
img.save(buffered, format="JPEG")
img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

model = "gpt-5"

# 测试1：纯文本
print(f"\n{'=' * 80}")
print(f"测试1: {model} - 纯文本")
print("=" * 80)
try:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "请说'你好'"}],
        max_tokens=1000
    )
    print(f"✅ 响应: {response.choices[0].message.content if response.choices else '空'}")
except Exception as e:
    print(f"❌ 失败: {e}")

# 测试2：文本 + 图像
print(f"\n{'=' * 80}")
print(f"测试2: {model} - 文本 + 图像")
print("=" * 80)
try:
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "描述这张图"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            ]
        }],
        max_tokens=1000,
        timeout=30
    )
    print(f"✅ Choices数量: {len(response.choices) if response.choices else 0}")
    if response.choices:
        print(f"✅ 响应: {response.choices[0].message.content}")
    else:
        print(f"⚠️  响应为空")
        print(f"   完整响应: {response}")
except Exception as e:
    print(f"❌ 失败: {e}")

print(f"\n{'=' * 80}")
