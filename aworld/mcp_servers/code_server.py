"""
Code Generation and Execution MCP Server

This module provides MCP server functionality for generating and executing code
using language models. It supports multiple programming languages and provides
tools for code generation, execution, and analysis.

Key features:
- Generate code based on user prompts and context
- Execute code in a sandboxed environment
- Analyze generated code for structure and content

Main functions:
- mcpgenerate: Generates code using language models
- mcpexecute: Executes code in a sandbox environment
"""

import json
import os
import subprocess
import tempfile
import traceback
import uuid
from collections import Counter
from typing import Dict, List, Optional

import pyglove as pg
from pydantic import BaseModel, Field

from aworld.logs.util import logger
from aworld.mcp_servers.abc.base import MCPServerBase, mcp
from aworld.mcp_servers.utils import (
    get_llm_config_from_os_environ,
    parse_port,
    run_mcp_server,
)
from aworld.models.llm import get_llm_model


class CodeGenerationResult(BaseModel):
    """Model representing the result of code generation"""

    code: str
    language: str
    explanation: Optional[str] = None


class CodeExecutionResult(BaseModel):
    """Model representing the result of code execution"""

    stdout: str
    stderr: str
    exit_code: int
    execution_time: float  # in seconds
    success: bool


class CodeServer(MCPServerBase):
    """
    Code Server class for generating and executing code.

    This class provides methods for generating code using language models and
    executing code in a sandboxed environment with various programming languages.
    """

    _instance = None
    _name = "code"
    _llm = None
    _llm_config = None

    def __new__(cls):
        """Implement singleton pattern"""
        if cls._instance is None:
            cls._instance = super(CodeServer, cls).__new__(cls)
            cls._instance._init_server()
        return cls._instance

    def _init_server(self):
        """Initialize the code server"""
        self._llm_config = get_llm_config_from_os_environ(
            llm_model_name="gpt-4o", server_name="Code Server"
        )
        self._llm = get_llm_model(self._llm_config)
        logger.info("CodeServer initialized")

    @classmethod
    def get_instance(cls):
        """Get the singleton instance of CodeServer"""
        if cls._instance is None:
            return cls()
        return cls._instance

    @staticmethod
    def _analyze_code(code: str, language: str) -> str:
        """Analyze the generated code to provide an explanation

        Args:
            code: The generated code to analyze
            language: The programming language of the code

        Returns:
            A string containing the explanation of the code
        """
        # Use PyGlove to analyze the code structure
        try:
            # Create a symbolic representation of the code
            code_sym = pg.Dict(
                code=code,
                language=language,
                lines=len(code.split("\n")),
                characters=len(code),
            )

            # Simple analysis based on language
            if language.lower() == "python":
                imports = [
                    line.strip()
                    for line in code.split("\n")
                    if line.strip().startswith(("import ", "from "))
                ]
                functions = len(
                    [
                        line
                        for line in code.split("\n")
                        if line.strip().startswith(("def ", "class "))
                    ]
                )
                code_sym.update(imports=imports, functions=functions)

            # Generate explanation
            explanation = f"This {language} code consists of {code_sym.lines} lines and {code_sym.characters} characters. "

            if language.lower() == "python" and code_sym.imports:
                explanation += f"It imports {len(code_sym.imports)} modules/packages: {', '.join(code_sym.imports)}. "
                explanation += f"The code defines approximately {code_sym.functions} functions/classes."

            return explanation

        except Exception as e:
            logger.warning(f"Code analysis error: {traceback.format_exc()}")
            line_count = len(code.split("\n"))
            return f"Generated {language} code with {line_count} lines."

    @mcp
    @classmethod
    def generate_code(
        cls,
        prompt: str = Field(..., description="Description of the code to be generated"),
        language: str = Field(
            "python",
            description="Programming language (default: python). Choices: python, javascript, bash",
        ),
        context: Optional[str] = Field(
            None, description="Additional context or existing code"
        ),
    ) -> str:
        """Generate code using LLM based on the provided prompt and context, ONLY support python, javascript, and bash languages

        Args:
            prompt: Description of the code to be generated
            language: Programming language for the generated code
            context: Additional context or existing code to consider

        Returns:
            JSON string with the generated code and explanation
        """
        try:
            # Handle Field objects if they're passed directly
            if hasattr(prompt, "default") and not isinstance(prompt, str):
                prompt = prompt.default

            if hasattr(language, "default") and not isinstance(language, str):
                language = language.default

            if (
                hasattr(context, "default")
                and not isinstance(context, str)
                and context is not None
            ):
                context = context.default

            # Get the singleton instance and ensure server is initialized
            instance = cls.get_instance()
            llm = instance._llm

            # Prepare the prompt for code generation
            system_prompt = (
                f"You are an expert {language} programmer. Generate clean, efficient, and well-documented "
                f"{language} code based on the user's requirements. Include comments explaining key parts of the code. "
                "Return ONLY the code without any additional text or explanations outside the code."
            )

            user_prompt = prompt
            if context:
                user_prompt = (
                    f"Context:\n```{language}\n{context}\n```\n\nRequirements: {prompt}"
                )

            # Generate code using LLM
            logger.info(f"Generating {language} code for: {prompt[:100]}...")

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            response = llm.completion(
                messages=messages,
                temperature=0.1,
                # max_tokens=16 * 1024,
            )
            instance._token_usage = dict(Counter(response.usage) + Counter(instance._token_usage))

            generated_code = response.content.strip() if response.content else ""

            # Extract code from markdown code blocks if present
            if generated_code.startswith("```") and generated_code.endswith("```"):
                # Remove the language identifier line and closing backticks
                code_lines = generated_code.split("\n")
                if len(code_lines) > 2:
                    generated_code = "\n".join(code_lines[1:-1])

            # Generate explanation directly without using pg.explore
            explanation = cls._analyze_code(generated_code, language)

            # Create result
            result = CodeGenerationResult(
                code=generated_code, language=language, explanation=explanation
            )

            return result.model_dump_json()

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Code generation error: {traceback.format_exc()}")
            return json.dumps({"error": error_msg})

    @classmethod
    def execute_code(
        cls,
        code: str = Field(
            ..., description="Code to execute in the sandbox environment"
        ),
        language: str = Field(
            "python",
            description="Programming language of the code (default: python). Choices: python, javascript, bash",
        ),
        timeout: int = Field(
            30, description="Maximum execution time in seconds (default: 30)"
        ),
        args: List[str] = Field(
            [], description="Command line arguments to pass to the program"
        ),
        env_vars: Dict[str, str] = Field(
            {}, description="Environment variables for execution"
        ),
    ) -> str:
        """Execute code in a sandbox environment and return the results, ONLY support python, javascript, and bash languages

        Args:
            code: Code to execute
            language: Programming language of the code
            timeout: Maximum execution time in seconds
            args: Command line arguments to pass to the program
            env_vars: Environment variables for execution

        Returns:
            JSON string with execution results
        """
        try:
            # Handle Field objects if they're passed directly
            if hasattr(code, "default") and not isinstance(code, str):
                code = code.default

            if hasattr(language, "default") and not isinstance(language, str):
                language = language.default

            if hasattr(timeout, "default") and not isinstance(timeout, int):
                timeout = timeout.default

            if hasattr(args, "default") and not isinstance(args, list):
                args = args.default

            if hasattr(env_vars, "default") and not isinstance(env_vars, dict):
                env_vars = env_vars.default

            # Get the singleton instance (not needed for this method but included for consistency)
            instance = cls.get_instance()

            # Create a temporary directory for execution
            with tempfile.TemporaryDirectory() as temp_dir:
                # Prepare execution based on language
                if language.lower() == "python":
                    file_extension = ".py"
                    command_prefix = ["python"]
                elif language.lower() == "javascript":
                    file_extension = ".js"
                    command_prefix = ["node"]
                elif language.lower() == "bash" or language.lower() == "shell":
                    file_extension = ".sh"
                    command_prefix = ["bash"]
                else:
                    raise ValueError(f"Unsupported language: {language}")

                # Create a unique filename
                filename = f"code_{uuid.uuid4().hex}{file_extension}"
                file_path = os.path.join(temp_dir, filename)

                # Write code to file
                with open(file_path, "w") as f:
                    f.write(code)

                # Make the file executable if it's a shell script
                if language.lower() in ["bash", "shell"]:
                    os.chmod(file_path, 0o755)

                # Prepare command
                command = command_prefix + [file_path] + args

                # Prepare environment
                execution_env = os.environ.copy()
                execution_env.update(env_vars)

                # Execute code in sandbox
                logger.info(f"Executing {language} code in sandbox...")

                import time

                start_time = time.time()

                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=execution_env,
                    cwd=temp_dir,
                    text=True,
                )

                try:
                    stdout, stderr = process.communicate(timeout=timeout)
                    exit_code = process.returncode
                    success = exit_code == 0
                except subprocess.TimeoutExpired:
                    process.kill()
                    stdout, stderr = process.communicate()
                    exit_code = -1
                    stderr += "\nExecution timed out"
                    success = False

                execution_time = time.time() - start_time

                # Create result
                result = CodeExecutionResult(
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=exit_code,
                    execution_time=execution_time,
                    success=success,
                )

                return result.model_dump_json()

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Code execution error: {traceback.format_exc()}")
            return json.dumps({"error": error_msg})


if __name__ == "__main__":
    port = parse_port()

    code_server = CodeServer.get_instance()
    logger.info("CodeServer initialized and ready to handle requests")

    run_mcp_server(
        "Code Generation and Execution Server",
        funcs=[code_server.generate_code, code_server.execute_code],
        port=port,
    )
