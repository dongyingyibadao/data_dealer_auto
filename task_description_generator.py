"""
使用Qwen/Deepseek LLM生成任务描述
"""
import json
from typing import List, Dict, Optional
import requests
from abc import ABC, abstractmethod
import base64
import io
try:
    from PIL import Image
except ImportError:
    import sys
    print("Installing Pillow...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
    from PIL import Image
import numpy as np
import torch


class LLMProvider(ABC):
    """LLM提供者基类"""
    
    @abstractmethod
    def generate_task_description(self, 
                                 action_type: str,
                                 original_task: str,
                                 context: Dict) -> str:
        """
        生成任务描述
        
        Args:
            action_type: 'pick' 或 'place'
            original_task: 原始任务描述
            context: 额外上下文信息 (包含图像等)
            
        Returns:
            生成的任务描述
        """
        pass


class GPTVLM(LLMProvider):
    """
    使用GPT-4o (VLM) 生成任务描述
    """
    
    def __init__(self, api_key: str = None, api_base: str = None, api_version: str = None, model: str = "gpt-4o", fast_mode: bool = False):
        self.api_key = api_key
        self.api_base = api_base
        self.api_version = api_version
        self.model = model
        self.fast_mode = fast_mode  # 快速模式：仅使用2帧（cam1首尾帧）
        self.available = api_key is not None
        
    def _encode_image(self, image_data):
        """将图像转换为base64字符串"""
        # 1. 处理 Tensor -> Numpy
        if hasattr(image_data, 'cpu'):
            image_data = image_data.cpu()
        if hasattr(image_data, 'numpy'):
            image_data = image_data.numpy()
            
        # 2. 处理 Numpy 数组
        if isinstance(image_data, np.ndarray):
            # 确保是 HWC 格式
            # 假设: 如果 shape[0] 是 3，且后面两个维度比 3 大，则是 CHW
            if image_data.ndim == 3 and image_data.shape[0] == 3 and image_data.shape[1] > 3 and image_data.shape[2] > 3:
                image_data = image_data.transpose(1, 2, 0)
            
            # 确保值在 0-255 之间且为 uint8
            if image_data.dtype != np.uint8:
                if image_data.max() <= 1.0:
                    image_data = (image_data * 255).astype(np.uint8)
                else:
                    image_data = image_data.astype(np.uint8)
            
            img = Image.fromarray(image_data)
            
        elif isinstance(image_data, Image.Image):
            img = image_data
        else:
            # 尝试作为 PIL Image 打开 (如果是路径字符串)
            try:
                img = Image.open(image_data)
            except:
                raise ValueError(f"Unsupported image type: {type(image_data)}")
            
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def generate_task_description(self, 
                                 action_type: str,
                                 original_task: str,
                                 context: Dict = None) -> str:
        if not self.available:
            print("⚠️  GPT API Key未提供，无法使用VLM")
            return f"{action_type} object"
            
        try:
            import openai
            
            # 检查是否使用 Azure OpenAI
            if self.api_version:
                from openai import AzureOpenAI
                client = AzureOpenAI(
                    api_key=self.api_key,
                    azure_endpoint=self.api_base,
                    api_version=self.api_version
                )
            else:
                client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url=self.api_base
                )
            
            # 检查是否有两个摄像头的图像
            first_cam1 = context.get('first_frame_cam1')
            last_cam1 = context.get('last_frame_cam1')
            key_cam1 = context.get('key_frame_cam1')
            first_cam2 = context.get('first_frame_cam2')
            last_cam2 = context.get('last_frame_cam2')
            key_cam2 = context.get('key_frame_cam2')
            
            # 快速模式：只需要cam1的首尾帧
            if self.fast_mode:
                if first_cam1 is None or last_cam1 is None:
                    print("⚠️  GPT VLM 缺少图像数据")
                    return f"{action_type} object"
                
                # 仅编码cam1的首尾两帧
                first_cam1_b64 = self._encode_image(first_cam1)
                last_cam1_b64 = self._encode_image(last_cam1)
                
                # 构建图像说明（快速模式）
                cam_info = "我提供了摄像头的图像"
                img_order = """
图像顺序：
1. 首帧（动作开始前）
2. 尾帧（动作完成后）"""
            else:
                # 精细模式：使用所有帧
                if first_cam1 is None or last_cam1 is None or key_cam1 is None:
                    print("⚠️  GPT VLM 缺少图像数据")
                    return f"{action_type} object"
                
                # 编码所有图像
                first_cam1_b64 = self._encode_image(first_cam1)
                last_cam1_b64 = self._encode_image(last_cam1)
                key_cam1_b64 = self._encode_image(key_cam1)
                
                # 如果有第二个摄像头的图像，也编码
                has_cam2 = first_cam2 is not None and last_cam2 is not None and key_cam2 is not None
                if has_cam2:
                    first_cam2_b64 = self._encode_image(first_cam2)
                    last_cam2_b64 = self._encode_image(last_cam2)
                    key_cam2_b64 = self._encode_image(key_cam2)
                else:
                    has_cam2 = False
                
                # 构建图像说明（精细模式）
                cam_info = "我提供了来自两个不同视角摄像头的图像" if has_cam2 else "我提供了摄像头的图像"
                img_order = """
图像顺序：
1-3. Camera 1 (整体场景视角): 首帧、关键帧(动作发生时刻)、尾帧
4-6. Camera 2 (操作细节视角): 首帧、关键帧(动作发生时刻)、尾帧""" if has_cam2 else """
图像顺序：
1. 首帧
2. 关键帧(动作发生时刻)
3. 尾帧"""
            
            # 根据模式构建不同的prompt
            if self.fast_mode:
                prompt = f"""
原始任务描述: "{original_task}"
动作类型: "{action_type}" (pick=抓取物体, place=放置物体)

{cam_info}，帮助你理解这个动作片段。{img_order}

重要说明：
1. **每个动作片段只操作一个物体** - 机械臂每次只能抓取或放置一个物体
2. **对比首尾帧的变化** - 注意哪个物体的位置发生了改变
3. **注意物体描述的完整性** - 例如"yellow and white mug"是一个黄白相间的杯子，不是两个杯子

观察要点：
- 对比首尾两帧，哪个物体的位置发生了变化？
- 该物体的完整描述是什么？(包括颜色、形状等特征)

输出格式：
- 格式必须是: "pick [object]" 或 "place [object] [location]"
- [object]必须是完整的物体描述(如: "white mug", "yellow and white mug")
- 只返回一行描述，不要其他内容

示例：
原始任务: "put the yellow and white mug on the plate"
动作类型: "pick"
正确输出: pick the yellow and white mug
"""
            else:
                prompt = f"""
原始任务描述: "{original_task}"
动作类型: "{action_type}" (pick=抓取物体, place=放置物体)

{cam_info}，帮助你理解这个动作片段。{img_order}

重要说明：
1. **每个动作片段只操作一个物体** - 机械臂每次只能抓取或放置一个物体
2. **Camera 1提供整体场景**，Camera 2提供操作细节和近距离视角
3. **仔细识别物体特征** - 注意物体的颜色、形状、纹理等特征
4. **注意物体描述的完整性** - 例如"yellow and white mug"是一个黄白相间的杯子，不是两个杯子

观察要点：
- 对比关键帧前后，哪个物体的位置发生了变化？
- 机械臂夹爪接触或操作的是哪个具体物体？
- 该物体的完整描述是什么？(包括颜色、图案等特征)

输出格式：
- 格式必须是: "pick [object]" 或 "place [object] [location]"
- [object]必须是完整的物体描述(如: "white mug", "yellow and white mug", "chocolate pudding")
- 只返回一行描述，不要其他内容

示例：
原始任务: "put the yellow and white mug on the plate"
动作类型: "pick"
正确输出: pick the yellow and white mug

原始任务: "put the red bowl and the blue cup on the table"  
动作类型: "place"
(观察图像后发现操作的是蓝色杯子)
正确输出: place the blue cup on the table
"""
            
            # 构建图像内容列表
            image_contents = [{"type": "text", "text": prompt}]
            
            if self.fast_mode:
                # 快速模式：仅上传cam1的首尾两帧
                image_contents.extend([
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{first_cam1_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{last_cam1_b64}"}},
                ])
            else:
                # 精细模式：上传cam1的三帧
                image_contents.extend([
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{first_cam1_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{key_cam1_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{last_cam1_b64}"}},
                ])
                
                # 如果有第二个摄像头的图像，添加
                if has_cam2:
                    image_contents.extend([
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{first_cam2_b64}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{key_cam2_b64}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{last_cam2_b64}"}},
                    ])

            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": image_contents}],
                max_tokens=50
            )
            
            # 调试信息：打印响应
            if response.choices and len(response.choices) > 0:
                message = response.choices[0].message
                if message.content is None:
                    print(f"⚠️  GPT 返回了空内容")
                    print(f"   Response: {response}")
                    print(f"   Message: {message}")
                    return f"{action_type} object"
                return message.content.strip()
            else:
                print(f"⚠️  GPT 返回了空的 choices")
                print(f"   Response: {response}")
                return f"{action_type} object"
            
        except Exception as e:
            print(f"⚠️  GPT VLM 调用失败: {e}")
            import traceback
            traceback.print_exc()
            return f"{action_type} object"


class QwenLLM(LLMProvider):
    """
    使用阿里Qwen模型（通过兼容OpenAI的API）
    """
    
    def __init__(self, api_key: str = None, api_base: str = None, model: str = "qwen-turbo"):
        """
        初始化Qwen LLM
        
        Args:
            api_key: API密钥
            api_base: API基础URL
            model: 模型名称
        """
        self.api_key = api_key
        self.api_base = api_base or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.model = model
        self.available = api_key is not None
    
    def generate_task_description(self, 
                                 action_type: str,
                                 original_task: str,
                                 context: Dict = None) -> str:
        """
        使用Qwen生成任务描述
        """
        if not self.available:
            return self._generate_local(action_type, original_task, context)
        
        try:
            import openai
            client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.api_base
            )
            
            prompt = self._build_prompt(action_type, original_task, context)
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个机器人任务描述生成器。根据原始任务和操作类型，生成简洁的任务描述。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=100
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"⚠️  Qwen API调用失败: {e}，使用本地方法生成")
            return self._generate_local(action_type, original_task, context)
    
    @staticmethod
    def _build_prompt(action_type: str, original_task: str, context: Dict = None) -> str:
        """构建提示词"""
        return f"""
根据以下信息，生成一个简洁的机器人任务描述：

原始任务: {original_task}
操作类型: {'夹爪关闭（抓取）' if action_type == 'pick' else '夹爪打开（放置）'}

要求：
1. 保留原始任务中关键的物体和位置信息
2. 根据操作类型（pick/place）生成对应的动词
3. Pick操作使用"pick up"或"grab"，Place操作使用"put"或"place"
4. 生成的描述应该简洁，不超过20个单词
5. 仅返回生成的描述，不需要其他内容

生成的任务描述：
"""
    
    @staticmethod
    def _generate_local(action_type: str, original_task: str, context: Dict = None) -> str:
        """本地生成任务描述（无API时使用）"""
        # 提取物体和位置信息
        words = original_task.lower().split()
        
        # 找到关键词
        object_words = []
        location_words = []
        
        # 简单的关键词提取
        prepositions = ['the', 'on', 'in', 'at', 'to', 'from', 'under', 'above', 'next', 'between']
        for i, word in enumerate(words):
            if word not in prepositions and word not in ['and', 'or', 'put', 'pick', 'up', 'open', 'close']:
                if i < len(words) - 1 and words[i + 1] not in prepositions:
                    object_words.append(word)
                elif i == len(words) - 1:
                    object_words.append(word)
        
        # 生成描述
        if action_type == 'pick':
            verb = 'pick up'
            if object_words:
                obj = object_words[0]
                return f"{verb} the {obj}"
            else:
                return f"{verb} the object"
        else:  # place
            verb = 'put'
            if len(object_words) >= 2:
                obj = object_words[0]
                loc = object_words[1]
                return f"{verb} the {obj} on the {loc}"
            elif len(object_words) >= 1:
                obj = object_words[0]
                return f"{verb} the {obj}"
            else:
                return f"{verb} the object"


class DeepseekLLM(LLMProvider):
    """
    使用Deepseek模型（通过OpenAI兼容API）
    """
    
    def __init__(self, api_key: str = None, api_base: str = None, model: str = "deepseek-chat"):
        """
        初始化Deepseek LLM
        
        Args:
            api_key: API密钥
            api_base: API基础URL
            model: 模型名称
        """
        self.api_key = api_key
        self.api_base = api_base or "https://api.deepseek.com/beta"
        self.model = model
        self.available = api_key is not None
    
    def generate_task_description(self, 
                                 action_type: str,
                                 original_task: str,
                                 context: Dict = None) -> str:
        """
        使用Deepseek生成任务描述
        """
        if not self.available:
            return self._generate_local(action_type, original_task, context)
        
        try:
            import openai
            client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.api_base
            )
            
            prompt = self._build_prompt(action_type, original_task, context)
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个机器人任务描述生成器。根据原始任务和操作类型，生成简洁的任务描述。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=100
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"⚠️  Deepseek API调用失败: {e}，使用本地方法生成")
            return self._generate_local(action_type, original_task, context)
    
    @staticmethod
    def _build_prompt(action_type: str, original_task: str, context: Dict = None) -> str:
        """构建提示词"""
        return f"""
根据以下信息，生成一个简洁的机器人任务描述：

原始任务: {original_task}
操作类型: {'夹爪关闭（抓取）' if action_type == 'pick' else '夹爪打开（放置）'}

要求：
1. 保留原始任务中关键的物体和位置信息
2. 根据操作类型（pick/place）生成对应的动词
3. Pick操作使用"pick up"或"grab"，Place操作使用"put"或"place"
4. 生成的描述应该简洁，不超过20个单词
5. 仅返回生成的描述，不需要其他内容

生成的任务描述：
"""
    
    @staticmethod
    def _generate_local(action_type: str, original_task: str, context: Dict = None) -> str:
        """本地生成任务描述（无API时使用）"""
        # 提取物体和位置信息
        words = original_task.lower().split()
        
        # 找到关键词
        object_words = []
        prepositions = ['the', 'on', 'in', 'at', 'to', 'from', 'under', 'above', 'next', 'between']
        
        for i, word in enumerate(words):
            if word not in prepositions and word not in ['and', 'or', 'put', 'pick', 'up', 'open', 'close']:
                if i < len(words) - 1 and words[i + 1] not in prepositions:
                    object_words.append(word)
                elif i == len(words) - 1:
                    object_words.append(word)
        
        # 生成描述
        if action_type == 'pick':
            verb = 'pick up'
            if object_words:
                obj = object_words[0]
                return f"{verb} the {obj}"
            else:
                return f"{verb} the object"
        else:  # place
            verb = 'put'
            if len(object_words) >= 2:
                obj = object_words[0]
                loc = object_words[1]
                return f"{verb} the {obj} on the {loc}"
            elif len(object_words) >= 1:
                obj = object_words[0]
                return f"{verb} the {obj}"
            else:
                return f"{verb} the object"


class TaskDescriptionGenerator:
    """
    任务描述生成器
    """
    
    def __init__(self, provider: str = "local", **kwargs):
        """
        初始化生成器
        
        Args:
            provider: 'qwen', 'deepseek', 'gpt', 或 'local'
            **kwargs: 传递给LLM提供者的参数（包括fast_mode）
        """
        # 过滤掉 None 值的参数，避免传递给构造函数
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        
        if provider.lower() == 'qwen':
            self.llm = QwenLLM(**kwargs)
        elif provider.lower() == 'deepseek':
            self.llm = DeepseekLLM(**kwargs)
        elif provider.lower() == 'gpt':
            self.llm = GPTVLM(**kwargs)
        else:
            self.llm = QwenLLM()  # 默认使用本地方法
    
    def generate_descriptions(self, 
                            frame_ranges: List[Dict],
                            dataset = None,
                            cache: Dict = None) -> List[Dict]:
        """
        为所有帧范围生成任务描述
        
        Args:
            frame_ranges: 帧范围列表
            dataset: LeRobot数据集 (用于VLM获取图像)
            cache: 缓存已生成的描述
            
        Returns:
            添加了new_task字段的帧范围列表
        """
        if cache is None:
            cache = {}
        
        result = []
        
        print(f"🤖 使用{self.llm.__class__.__name__}生成任务描述...")
        
        for i, frame_range in enumerate(frame_ranges):
            if i % 10 == 0:
                print(f"  进度: {i}/{len(frame_ranges)}")
            
            # 创建缓存键
            # 注意：对于VLM，如果只用action_type和task做key，会忽略图像差异。
            # 如果是VLM，我们可能不应该使用简单的缓存，或者应该包含keyframe_index
            if isinstance(self.llm, GPTVLM):
                cache_key = f"{frame_range['action_type']}_{frame_range['task']}_{frame_range['keyframe_index']}"
            else:
                cache_key = f"{frame_range['action_type']}_{frame_range['task']}"
            
            if cache_key in cache:
                new_task = cache[cache_key]
            else:
                # 准备上下文
                context = {'episode_index': frame_range['episode_index']}
                
                # 如果是VLM且提供了数据集，获取图像
                if isinstance(self.llm, GPTVLM) and dataset is not None:
                    try:
                        start_idx = int(frame_range['frame_start'])
                        end_idx = int(frame_range['frame_end']) - 1 # frame_end is exclusive
                        key_idx = int(frame_range['keyframe_index'])
                        
                        # 获取首尾帧和关键帧 (两个摄像头)
                        # LeRobot dataset returns dict with 'observation.images.image' and 'observation.images.image2'
                        first_item = dataset[start_idx]
                        last_item = dataset[end_idx]
                        key_item = dataset[key_idx]
                        
                        # Cam1 图像
                        context['first_frame_cam1'] = first_item['observation.images.image']
                        context['last_frame_cam1'] = last_item['observation.images.image']
                        context['key_frame_cam1'] = key_item['observation.images.image']
                        
                        # Cam2 图像
                        context['first_frame_cam2'] = first_item['observation.images.image2']
                        context['last_frame_cam2'] = last_item['observation.images.image2']
                        context['key_frame_cam2'] = key_item['observation.images.image2']
                    except Exception as e:
                        print(f"⚠️  获取图像失败: {e}")
                        import traceback
                        traceback.print_exc()
                
                # 生成新的任务描述
                new_task = self.llm.generate_task_description(
                    action_type=frame_range['action_type'],
                    original_task=frame_range['task'],
                    context=context
                )
                cache[cache_key] = new_task
            
            # 添加到结果
            range_with_desc = frame_range.copy()
            range_with_desc['new_task'] = new_task
            result.append(range_with_desc)
        
        print(f"✓ 任务描述生成完成")
        return result


if __name__ == '__main__':
    # 测试本地生成
    generator = TaskDescriptionGenerator(provider='local')
    
    test_ranges = [
        {
            'action_type': 'pick',
            'task': 'put both moka pots on the stove',
            'episode_index': 376
        },
        {
            'action_type': 'place',
            'task': 'put both moka pots on the stove',
            'episode_index': 376
        }
    ]
    
    results = generator.generate_descriptions(test_ranges)
    for r in results:
        print(f"{r['action_type']}: {r['task']} -> {r['new_task']}")
