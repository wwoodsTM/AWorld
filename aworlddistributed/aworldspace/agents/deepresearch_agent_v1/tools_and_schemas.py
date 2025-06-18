from typing import List, TypeVar, Type
from pydantic import BaseModel, Field, ValidationError
import json

class SearchQueryList(BaseModel):
    query: List[str] = Field(
        description="A list of search queries to be used for web research."
    )
    rationale: str = Field(
        description="A brief explanation of why these queries are relevant to the research topic."
    )

class Reflection(BaseModel):
    is_sufficient: bool = Field(
        description="Whether the provided summaries are sufficient to answer the user's question."
    )
    knowledge_gap: str = Field(
        description="A description of what information is missing or needs clarification."
    )
    follow_up_queries: List[str] = Field(
        description="A list of follow-up queries to address the knowledge gap."
    )

class AworldSearch(BaseModel):
    title: str = Field(
        description="title"
    )
    docAbstract: str = Field(
        description="docAbstract"
    )
    url: str = Field(
        description="url"
    )
    doc: str = Field(
        description="doc"
    )


# 定义泛型类型
T = TypeVar('T', bound=BaseModel)

def parse_json_to_model(json_data: str, model_class: Type[T]) -> T:
    """
    JSON解析工具方法 - 支持处理大模型返回的带代码块标记的JSON

    Args:
        json_data (str): JSON字符串（支持带有```json```代码块标记的格式）
        model_class (Type[T]): Pydantic模型类类型

    Returns:
        T: 解析后的模型实例

    Raises:
        json.JSONDecodeError: 当JSON格式无效时
        ValidationError: 当数据不符合模型结构时

    Example:
        >>> # 普通JSON字符串
        >>> json_str = '{"query": ["test query"], "rationale": "test rationale"}'
        >>> result = parse_json_to_model(json_str, SearchQueryList)

        >>> # 带代码块标记的JSON（大模型常见返回格式）
        >>> json_with_blocks = '''```json
        ... {"query": ["test"], "rationale": "test"}
        ... ```'''
        >>> result = parse_json_to_model(json_with_blocks, SearchQueryList)
    """
    try:
        # 清理可能的代码块标记
        cleaned_json = json_data.strip()

        # 移除开头的代码块标记
        if cleaned_json.startswith('```json'):
            cleaned_json = cleaned_json[7:]  # 移除 '```json'
        elif cleaned_json.startswith('```'):
            cleaned_json = cleaned_json[3:]   # 移除 '```'

        # 移除结尾的```标记
        if cleaned_json.endswith('```'):
            cleaned_json = cleaned_json[:-3]

        # 再次清理空白字符
        cleaned_json = cleaned_json.strip()

        # 解析JSON字符串为字典或列表
        parsed_data = json.loads(cleaned_json)

        # 如果解析结果是列表，取第一个元素
        if isinstance(parsed_data, list):
            if len(parsed_data) == 0:
                raise ValueError(f"Empty list cannot be converted to {model_class.__name__}")
            data_dict = parsed_data[0]
        elif isinstance(parsed_data, dict):
            data_dict = parsed_data
        else:
            raise ValueError(f"Parsed data must be a dict or list, got {type(parsed_data)}")

        # 使用Pydantic模型验证并创建实例
        return model_class(**data_dict)

    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON format: {e.msg}", e.doc, e.pos)
    except ValidationError as e:
        raise ValidationError(f"Data validation failed for {model_class.__name__}: {e}")
    except Exception as e:
        raise Exception(f"Unexpected error while parsing JSON to {model_class.__name__}: {str(e)}")

def parse_json_to_model_list(json_data: str, model_class: Type[T]) -> List[T]:
    """
    JSON解析工具方法 - 专门处理JSON列表，返回模型实例列表

    Args:
        json_data (str): JSON字符串（应该是一个数组）
        model_class (Type[T]): Pydantic模型类类型

    Returns:
        List[T]: 解析后的模型实例列表

    Raises:
        json.JSONDecodeError: 当JSON格式无效时
        ValidationError: 当数据不符合模型结构时
        ValueError: 当JSON不是列表格式时

    Example:
        >>> json_str = '[{"title": "test", "doc": "content"}]'
        >>> results = parse_json_to_model_list(json_str, AworldSearch)
    """
    try:
        # 检查输入是否为空
        if not json_data or not json_data.strip():
            print(f"[WARNING] 输入的 JSON 数据为空")
            return []
            
        # 清理可能的代码块标记
        cleaned_json = json_data.strip()

        # 移除开头的代码块标记
        if cleaned_json.startswith('```json'):
            cleaned_json = cleaned_json[7:]  # 移除 '```json'
        elif cleaned_json.startswith('```'):
            cleaned_json = cleaned_json[3:]   # 移除 '```'

        # 移除结尾的```标记
        if cleaned_json.endswith('```'):
            cleaned_json = cleaned_json[:-3]

        # 再次清理空白字符
        cleaned_json = cleaned_json.strip()

        # 解析JSON字符串
        parsed_data = json.loads(cleaned_json)

        # 确保解析结果是列表
        if not isinstance(parsed_data, list):
            print(f"[WARNING] 期望 JSON 数组，但得到 {type(parsed_data)}")
            return []

        # 如果列表为空，直接返回空列表
        if len(parsed_data) == 0:
            print(f"[WARNING] JSON 数组为空")
            return []

        # 将列表中的每个字典转换为模型实例
        model_instances = []
        failed_count = 0
        
        for i, item in enumerate(parsed_data):
            if not isinstance(item, dict):
                print(f"[WARNING] 列表项 {i} 不是字典，跳过: {type(item)}")
                failed_count += 1
                continue
            
            try:
                model_instance = model_class(**item)
                model_instances.append(model_instance)
            except Exception as e:
                print(f"[WARNING] 创建模型实例 {i} 失败，跳过: {e}")
                failed_count += 1
                continue

        if failed_count > 0:
            print(f"[INFO] 成功创建 {len(model_instances)} 个模型实例，失败 {failed_count} 个")
        
        return model_instances

    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON 解析错误: {e}")
        return []  # 返回空列表而不是抛出异常
    except Exception as e:
        print(f"[ERROR] 解析过程中发生未知错误: {e}")
        return []  # 返回空列表而不是抛出异常