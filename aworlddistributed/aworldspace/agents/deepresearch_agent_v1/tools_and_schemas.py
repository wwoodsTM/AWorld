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

# 定义泛型类型
T = TypeVar('T', bound=BaseModel)

def parse_json_to_model(json_data: str, model_class: Type[T]) -> T:
    """
    JSON解析工具方法
    
    Args:
        json_data (str): JSON字符串
        model_class (Type[T]): Pydantic模型类类型
        
    Returns:
        T: 解析后的模型实例
        
    Raises:
        json.JSONDecodeError: 当JSON格式无效时
        ValidationError: 当数据不符合模型结构时
        
    Example:
        >>> json_str = '{"query": ["test query"], "rationale": "test rationale"}'
        >>> result = parse_json_to_model(json_str, SearchQueryList)
        >>> print(result.query)
        ['test query']
    """
    try:
        # 先解析JSON字符串为字典
        data_dict = json.loads(json_data)
        
        # 使用Pydantic模型验证并创建实例
        return model_class(**data_dict)
        
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON format: {e.msg}", e.doc, e.pos)
    except ValidationError as e:
        raise ValidationError(f"Data validation failed for {model_class.__name__}: {e}")
    except Exception as e:
        raise Exception(f"Unexpected error while parsing JSON to {model_class.__name__}: {str(e)}")