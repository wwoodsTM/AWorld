from typing import Any, Dict, List, Optional
import json


class ModelResponse:
    """
    Unified model response class for encapsulating responses from different LLM providers
    """
    
    def __init__(
        self,
        id: str,
        model: str,
        content: str = None,
        tool_calls: List[Dict[str, Any]] = None,
        function_call: Dict[str, Any] = None,
        usage: Dict[str, int] = None,
        error: str = None,
        raw_response: Any = None,
        message: Dict[str, Any] = None
    ):
        """
        Initialize ModelResponse object
        
        Args:
            id: Response ID
            model: Model name used
            content: Generated text content
            tool_calls: List of tool calls
            function_call: Function call information (compatible with legacy API)
            usage: Usage statistics (token counts, etc.)
            error: Error message (if any)
            raw_response: Original response object
            message: Complete message object, can be used for subsequent API calls
        """
        self.id = id
        self.model = model
        self.content = content
        self.tool_calls = tool_calls or []
        self.function_call = function_call
        self.usage = usage or {
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "total_tokens": 0
        }
        self.error = error
        self.raw_response = raw_response
        
        # If message is not provided, construct one from other fields
        if message is None:
            self.message = {
                "role": "assistant",
                "content": content
            }
            
            if tool_calls:
                self.message["tool_calls"] = tool_calls
                
            if function_call:
                self.message["function_call"] = function_call
        else:
            self.message = message
        
    @classmethod
    def from_openai_response(cls, response: Any) -> 'ModelResponse':
        """
        Create ModelResponse from OpenAI response object
        
        Args:
            response: OpenAI response object
            
        Returns:
            ModelResponse object
        """
        # Handle error cases
        if hasattr(response, 'error') or (isinstance(response, dict) and response.get('error')):
            error_msg = response.error if hasattr(response, 'error') else response.get('error', 'Unknown error')
            return cls(
                id=response.id if hasattr(response, 'id') else response.get('id', 'error'),
                model=response.model if hasattr(response, 'model') else response.get('model', 'unknown'),
                error=error_msg,
                raw_response=response
            )
        
        # Normal case
        message = None
        if hasattr(response, 'choices') and response.choices:
            message = response.choices[0].message
        elif isinstance(response, dict) and response.get('choices'):
            message = response['choices'][0].get('message', {})
        
        if not message:
            return cls(
                id=response.id if hasattr(response, 'id') else response.get('id', 'unknown'),
                model=response.model if hasattr(response, 'model') else response.get('model', 'unknown'),
                error="No message found in response",
                raw_response=response
            )
            
        # Extract usage information
        usage = {}
        if hasattr(response, 'usage'):
            usage = {
                "completion_tokens": response.usage.completion_tokens if hasattr(response.usage, 'completion_tokens') else 0,
                "prompt_tokens": response.usage.prompt_tokens if hasattr(response.usage, 'prompt_tokens') else 0,
                "total_tokens": response.usage.total_tokens if hasattr(response.usage, 'total_tokens') else 0
            }
        elif isinstance(response, dict) and response.get('usage'):
            usage = response['usage']
            
        # Build message object
        message_dict = {}
        if hasattr(message, '__dict__'):
            # Convert object to dictionary
            for key, value in message.__dict__.items():
                if not key.startswith('_'):
                    message_dict[key] = value
        elif isinstance(message, dict):
            message_dict = message
        else:
            # Extract common properties
            message_dict = {
                "role": "assistant",
                "content": message.content if hasattr(message, 'content') else None,
                "tool_calls": message.tool_calls if hasattr(message, 'tool_calls') else None,
                "function_call": message.function_call if hasattr(message, 'function_call') else None
            }
            
        # Create and return ModelResponse
        return cls(
            id=response.id if hasattr(response, 'id') else response.get('id', 'unknown'),
            model=response.model if hasattr(response, 'model') else response.get('model', 'unknown'),
            content=message.content if hasattr(message, 'content') else message.get('content'),
            tool_calls=message.tool_calls if hasattr(message, 'tool_calls') else message.get('tool_calls'),
            function_call=message.function_call if hasattr(message, 'function_call') else message.get('function_call'),
            usage=usage,
            raw_response=response,
            message=message_dict
        )
        
    @classmethod
    def from_openai_stream_chunk(cls, chunk: Any) -> 'ModelResponse':
        """
        Create ModelResponse from OpenAI stream response chunk
        
        Args:
            chunk: OpenAI stream chunk
            
        Returns:
            ModelResponse object
        """
        # Handle error cases
        if hasattr(chunk, 'error') or (isinstance(chunk, dict) and chunk.get('error')):
            error_msg = chunk.error if hasattr(chunk, 'error') else chunk.get('error', 'Unknown error')
            return cls(
                id=chunk.id if hasattr(chunk, 'id') else chunk.get('id', 'error'),
                model=chunk.model if hasattr(chunk, 'model') else chunk.get('model', 'unknown'),
                error=error_msg,
                raw_response=chunk
            )
        
        # Handle finish reason chunk (end of stream)
        if hasattr(chunk, 'choices') and chunk.choices and chunk.choices[0].finish_reason:
            return cls(
                id=chunk.id if hasattr(chunk, 'id') else chunk.get('id', 'unknown'),
                model=chunk.model if hasattr(chunk, 'model') else chunk.get('model', 'unknown'),
                content=None,
                raw_response=chunk,
                message={"role": "assistant", "content": None, "finish_reason": chunk.choices[0].finish_reason}
            )
            
        # Normal chunk with delta content
        message = None
        content = None
        tool_calls = None
        function_call = None
        
        if hasattr(chunk, 'choices') and chunk.choices:
            delta = chunk.choices[0].delta
            if hasattr(delta, 'content') and delta.content:
                content = delta.content
            if hasattr(delta, 'tool_calls') and delta.tool_calls:
                tool_calls = delta.tool_calls
            if hasattr(delta, 'function_call') and delta.function_call:
                function_call = delta.function_call
        elif isinstance(chunk, dict) and chunk.get('choices'):
            delta = chunk['choices'][0].get('delta', {})
            content = delta.get('content')
            tool_calls = delta.get('tool_calls')
            function_call = delta.get('function_call')
            
        # Create message object
        message = {
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls,
            "function_call": function_call,
            "is_chunk": True
        }
        
        # Create and return ModelResponse
        return cls(
            id=chunk.id if hasattr(chunk, 'id') else chunk.get('id', 'unknown'),
            model=chunk.model if hasattr(chunk, 'model') else chunk.get('model', 'unknown'),
            content=content,
            tool_calls=tool_calls,
            function_call=function_call,
            raw_response=chunk,
            message=message
        )
        
    @classmethod
    def from_anthropic_stream_chunk(cls, chunk: Any) -> 'ModelResponse':
        """
        Create ModelResponse from Anthropic stream response chunk
        
        Args:
            chunk: Anthropic stream chunk
            
        Returns:
            ModelResponse object
        """
        try:
            # Handle error cases
            if not chunk or (isinstance(chunk, dict) and chunk.get('error')):
                error_msg = chunk.get('error', 'Unknown error') if isinstance(chunk, dict) else 'Empty response'
                return cls.from_error(error_msg, "claude")
                
            # Handle stop reason (end of stream)
            if hasattr(chunk, 'stop_reason') and chunk.stop_reason:
                return cls(
                    id=chunk.id if hasattr(chunk, 'id') else 'unknown',
                    model=chunk.model if hasattr(chunk, 'model') else 'claude',
                    content=None,
                    raw_response=chunk,
                    message={"role": "assistant", "content": None, "stop_reason": chunk.stop_reason}
                )
                
            # Handle delta content
            content = None
            tool_calls = None
            function_call = None
            
            if hasattr(chunk, 'delta') and chunk.delta:
                delta = chunk.delta
                if hasattr(delta, 'text') and delta.text:
                    content = delta.text
                elif hasattr(delta, 'tool_use') and delta.tool_use:
                    tool_call = {
                        "id": f"call_{delta.tool_use.id}",
                        "type": "function",
                        "function": {
                            "name": delta.tool_use.name,
                            "arguments": delta.tool_use.input if isinstance(delta.tool_use.input, str) else json.dumps(delta.tool_use.input)
                        }
                    }
                    tool_calls = [tool_call]
                    function_call = {
                        "name": delta.tool_use.name,
                        "arguments": delta.tool_use.input if isinstance(delta.tool_use.input, str) else json.dumps(delta.tool_use.input)
                    }
                    
            # Create message object
            message = {
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls,
                "function_call": function_call,
                "is_chunk": True
            }
            
            # Create and return ModelResponse
            return cls(
                id=chunk.id if hasattr(chunk, 'id') else 'unknown',
                model=chunk.model if hasattr(chunk, 'model') else 'claude',
                content=content,
                tool_calls=tool_calls,
                function_call=function_call,
                raw_response=chunk,
                message=message
            )
            
        except Exception as e:
            return cls.from_error(f"Error processing Anthropic stream chunk: {str(e)}", "claude")
        
    @classmethod
    def from_anthropic_response(cls, response: Any) -> 'ModelResponse':
        """
        Create ModelResponse from Anthropic original response object
        
        Args:
            response: Anthropic response object
            
        Returns:
            ModelResponse object
        """
        try:
            # Handle error cases
            if not response or (isinstance(response, dict) and response.get('error')):
                error_msg = response.get('error', 'Unknown error') if isinstance(response, dict) else 'Empty response'
                return cls.from_error(error_msg, "claude")
                
            # Build message content
            message = {
                "content": None,
                "role": "assistant",
                "tool_calls": None,
                "function_call": None
            }

            if hasattr(response, 'content') and response.content:
                for content_block in response.content:
                    if content_block.type == "text":
                        message["content"] = content_block.text
                    elif content_block.type == "tool_use":
                        if message["tool_calls"] is None:
                            message["tool_calls"] = []

                        tool_call = {
                            "id": f"call_{content_block.id}",
                            "type": "function",
                            "function": {
                                "name": content_block.name,
                                "arguments": content_block.input if isinstance(content_block.input, str) else json.dumps(content_block.input)
                            }
                        }
                        message["tool_calls"].append(tool_call)

                        if len(message["tool_calls"]) == 1:
                            message["function_call"] = {
                                "name": content_block.name,
                                "arguments": content_block.input if isinstance(content_block.input, str) else json.dumps(content_block.input)
                            }
            else:
                message["content"] = ""
                
            # Extract usage information
            usage = {
                "completion_tokens": 0,
                "prompt_tokens": 0,
                "total_tokens": 0
            }
            
            if hasattr(response, 'usage'):
                if hasattr(response.usage, 'output_tokens'):
                    usage["completion_tokens"] = response.usage.output_tokens
                if hasattr(response.usage, 'input_tokens'):
                    usage["prompt_tokens"] = response.usage.input_tokens
                if hasattr(response.usage, 'input_tokens') and hasattr(response.usage, 'output_tokens'):
                    usage["total_tokens"] = response.usage.input_tokens + response.usage.output_tokens
                    
            # Create ModelResponse
            return cls(
                id=response.id if hasattr(response, 'id') else f"chatcmpl-anthropic-{hash(str(response)) & 0xffffffff:08x}",
                model=response.model if hasattr(response, 'model') else "claude",
                content=message["content"],
                tool_calls=message["tool_calls"],
                function_call=message["function_call"],
                usage=usage,
                raw_response=response,
                message=message
            )
        except Exception as e:
            return cls.from_error(f"Error processing Anthropic response: {str(e)}", "claude")
        
    @classmethod
    def from_error(cls, error_msg: str, model: str = "unknown") -> 'ModelResponse':
        """
        Create ModelResponse from error message
        
        Args:
            error_msg: Error message
            model: Model name
            
        Returns:
            ModelResponse object
        """
        return cls(
            id="error",
            model=model,
            error=error_msg,
            message={"role": "assistant", "content": f"Error: {error_msg}"}
        )
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert ModelResponse to dictionary representation
        
        Returns:
            Dictionary representation
        """
        return {
            "id": self.id,
            "model": self.model,
            "content": self.content,
            "tool_calls": self.tool_calls,
            "function_call": self.function_call,
            "usage": self.usage,
            "error": self.error,
            "message": self.message
        }
        
    def to_message(self) -> Dict[str, Any]:
        """
        Return message object that can be directly used for subsequent API calls
        
        Returns:
            Message object dictionary
        """
        return self.message
        
    def tool_calls_dump_json(self) -> List[Dict[str, Any]]:
        """
        Convert tool call objects to JSON format, handling OpenAI object types
        
        Returns:
            List[Dict[str, Any]]: Tool calls list in JSON format
        """
        if not self.tool_calls:
            return []
            
        result = []
        for tool_call in self.tool_calls:
            # If already in dictionary format, use directly
            if isinstance(tool_call, dict):
                result.append(tool_call)
                continue
                
            # Handle OpenAI object types
            tool_call_dict = {}
            
            # Extract id
            if hasattr(tool_call, 'id'):
                tool_call_dict['id'] = tool_call.id
            elif hasattr(tool_call, 'index'):
                tool_call_dict['id'] = f"call_{tool_call.index}"
                
            # Extract type
            if hasattr(tool_call, 'type'):
                tool_call_dict['type'] = tool_call.type
            else:
                tool_call_dict['type'] = 'function'
                
            # Extract function information
            if hasattr(tool_call, 'function'):
                function = tool_call.function
                function_dict = {}
                
                if hasattr(function, 'name'):
                    function_dict['name'] = function.name
                
                if hasattr(function, 'arguments'):
                    # Ensure arguments is a JSON string
                    if isinstance(function.arguments, str):
                        try:
                            # Try to parse JSON
                            json.loads(function.arguments)
                            function_dict['arguments'] = function.arguments
                        except:
                            # If not a valid JSON, convert to JSON string
                            function_dict['arguments'] = json.dumps(function.arguments)
                    else:
                        function_dict['arguments'] = json.dumps(function.arguments)
                        
                tool_call_dict['function'] = function_dict
                
            result.append(tool_call_dict)
            
        return result
        
    def to_json(self) -> str:
        """
        Convert ModelResponse to JSON string representation
        
        Returns:
            str: JSON string
        """
        json_dict = {
            "id": self.id,
            "model": self.model,
            "content": self.content,
            "tool_calls": self.tool_calls_dump_json(),
            "function_call": self._serialize_function_call(),
            "usage": self.usage,
            "error": self.error,
            "message": self._serialize_message()
        }
        
        return json.dumps(json_dict, ensure_ascii=False, indent=None)
        
    def _serialize_function_call(self) -> Dict[str, Any]:
        """
        Serialize function_call object
        
        Returns:
            Dict[str, Any]: Serialized function_call dictionary
        """
        if not self.function_call:
            return None
            
        # If already a dictionary, return directly
        if isinstance(self.function_call, dict):
            return self.function_call
            
        # Handle object types
        result = {}
        
        if hasattr(self.function_call, 'name'):
            result['name'] = self.function_call.name
            
        if hasattr(self.function_call, 'arguments'):
            # Ensure arguments is a JSON string
            if isinstance(self.function_call.arguments, str):
                try:
                    # Try to parse JSON to validate
                    json.loads(self.function_call.arguments)
                    result['arguments'] = self.function_call.arguments
                except:
                    # If not a valid JSON, convert to JSON string
                    result['arguments'] = json.dumps(self.function_call.arguments)
            else:
                result['arguments'] = json.dumps(self.function_call.arguments)
                
        return result
        
    def _serialize_message(self) -> Dict[str, Any]:
        """
        Serialize message object
        
        Returns:
            Dict[str, Any]: Serialized message dictionary
        """
        if not self.message:
            return {}
            
        result = {}
        
        # Copy basic fields
        for key, value in self.message.items():
            if key == 'tool_calls':
                # Handle tool_calls
                result[key] = self.tool_calls_dump_json()
            elif key == 'function_call':
                # Handle function_call
                result[key] = self._serialize_function_call()
            else:
                result[key] = value
                
        return result