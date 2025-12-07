"""
测试 Azure OpenAI API 密钥
测试不同模型的可用性
"""

from openai import AzureOpenAI

# 配置
API_KEY = "5ffef770a5b148c5920b7b16329e30fa"
AZURE_ENDPOINT = "https://gpt.yunstorm.com/"
API_VERSION = "2025-01-01-preview"

# 要测试的模型列表
MODELS_TO_TEST = [
    "gpt-5",
    "gpt-4.1",           # 你想测试的模型
    "gpt-4.1-mini",      # 可能的变体
    "gpt-4o",            # GPT-4o
    "gpt-4o-mini",       # GPT-4o mini
    "gpt-4",             # GPT-4
    "gpt-4-turbo",       # GPT-4 Turbo
    "gpt-35-turbo",      # GPT-3.5 Turbo (Azure 命名)
    "gpt-3.5-turbo",     # GPT-3.5 Turbo (标准命名)
]

def test_model(client, model_name: str) -> bool:
    """
    测试单个模型是否可用
    
    Args:
        client: AzureOpenAI 客户端
        model_name: 模型名称
        
    Returns:
        bool: 模型是否可用
    """
    try:
        print(f"\n🔄 正在测试模型: {model_name}")
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "你是一个擅长编程的计算机工作人士。"},
                {"role": "user", "content": "请告诉我现在的ChatGPT有哪些支持的api-model"}
            ],
            max_tokens=100,
            temperature=0.7
        )
        
        reply = response.choices[0].message.content
        model_used = response.model
        
        print(f"✅ 模型 {model_name} 可用!")
        print(f"   实际使用的模型: {model_used}")
        print(f"   回复: {reply}")
        return True
        
    except Exception as e:
        print(f"❌ 模型 {model_name} 不可用")
        print(f"   错误: {e}")
        return False


def main():
    print("=" * 60)
    print("Azure OpenAI API 测试脚本")
    print("=" * 60)
    print(f"API Endpoint: {AZURE_ENDPOINT}")
    print(f"API Version: {API_VERSION}")
    print(f"API Key: {API_KEY[:8]}...{API_KEY[-4:]}")
    print("=" * 60)
    
    # 创建客户端
    try:
        client = AzureOpenAI(
            api_key=API_KEY,
            azure_endpoint=AZURE_ENDPOINT,
            api_version=API_VERSION
        )
        print("✅ Azure OpenAI 客户端创建成功")
    except Exception as e:
        print(f"❌ 客户端创建失败: {e}")
        return
    
    # 测试所有模型
    available_models = []
    unavailable_models = []
    
    for model in MODELS_TO_TEST:
        if test_model(client, model):
            available_models.append(model)
        else:
            unavailable_models.append(model)
    
    # 总结
    print("\n" + "=" * 60)
    print("测试结果总结")
    print("=" * 60)
    
    if available_models:
        print(f"\n✅ 可用的模型 ({len(available_models)} 个):")
        for model in available_models:
            print(f"   - {model}")
    else:
        print("\n❌ 没有找到可用的模型")
    
    if unavailable_models:
        print(f"\n❌ 不可用的模型 ({len(unavailable_models)} 个):")
        for model in unavailable_models:
            print(f"   - {model}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
