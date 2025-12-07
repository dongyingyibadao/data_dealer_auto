"""
最小化测试：直接调用 Azure OpenAI 带图像
"""
from openai import AzureOpenAI
import base64
import io
from PIL import Image
import numpy as np

# 配置
API_KEY = "5ffef770a5b148c5920b7b16329e30fa"
API_BASE = "https://gpt.yunstorm.com/"
API_VERSION = "2025-01-01-preview"

print("=" * 80)
print("最小化测试：Azure OpenAI 带图像")
print("=" * 80)

# 创建客户端
client = AzureOpenAI(
    api_key=API_KEY,
    azure_endpoint=API_BASE,
    api_version=API_VERSION
)
print("✅ 客户端创建成功")

# 创建一个简单的测试图像
print("\n📸 创建测试图像...")
img = Image.new('RGB', (256, 256), color=(73, 109, 137))
buffered = io.BytesIO()
img.save(buffered, format="JPEG")
img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
print("✅ 图像编码成功")

# 测试不同模型
models = ["gpt-4o", "gpt-5"]

for model in models:
    print(f"\n{'=' * 80}")
    print(f"测试模型: {model}")
    print("=" * 80)
    
    try:
        print("🔄 发送请求...")
        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "请简单描述这张图片，一句话即可。"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]
            }],
            max_tokens=50
        )
        
        print(f"✅ 响应接收成功")
        print(f"   Response: {response}")
        print(f"   Response dict: {response.model_dump() if hasattr(response, 'model_dump') else 'N/A'}")
        print(f"   Choices length: {len(response.choices) if response.choices else 0}")
        
        if response.choices and len(response.choices) > 0:
            message = response.choices[0].message
            print(f"   Message: {message}")
            print(f"   Content: {message.content}")
            
            if message.content:
                print(f"   ✅ 内容: {message.content}")
            else:
                print(f"   ⚠️  Content 为 None")
        else:
            print(f"   ⚠️  Choices 为空")
            
    except Exception as e:
        print(f"❌ 失败: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'=' * 80}")
print("测试完成")
print("=" * 80)
