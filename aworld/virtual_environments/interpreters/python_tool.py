import sys
import ast
import re
import subprocess
from typing import Any, Dict, Tuple, List
from io import StringIO
from aworld.logs.util import logger
from aworld.config.conf import ToolConfig
from aworld.core.envs.tool_action import PythonToolAction
from aworld.core.common import ActionModel, Observation, ActionResult, Tools
from aworld.core.envs.env_tool import EnvTool, AgentInput, ToolFactory


@ToolFactory.register(name=Tools.PYTHON_EXECUTE.value, desc="python interpreter tool",
                      supported_action=PythonToolAction)
class PythonTool(EnvTool[Observation, List[ActionModel]]):

    def __init__(self,
                 conf: ToolConfig,
                 **kwargs) -> None:
        """
        Initialize the PythonExecutor
        Args:
            conf: tool config
            **kwargs: -
        Return:
            None
        """
        super(PythonTool, self).__init__(conf, **kwargs)
        self.type = "function"
        self.local_namespace = {}
        self.global_namespace = {}
        self.original_stdout = sys.stdout
        self.output_buffer = StringIO()
        self.step_finished = True
        self.installed_packages = set()

    def name(self):
        """
        Get the name of the tool
        Args:
            -
        Returns:
            str: tool name
        """
        return self.__class__.__name__

    def extract_imports(self, code: str) -> set:
        """
        Extract import statements
        Args:
            code: python code
        Returns:
            set: import statements
        """
        imports = set()

        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    # deal import xxx or import xxx as yyy
                    for name in node.names:
                        package_name = name.name.split('.')[0]
                        imports.add(package_name)

                elif isinstance(node, ast.ImportFrom):
                    # deal from xxx import yyy or from xxx.yyy import zzz
                    if node.module:
                        package_name = node.module.split('.')[0]
                        imports.add(package_name)

        except SyntaxError:
            import_pattern = r'^import\s+([\w\s,]+)|from\s+(\w+)'
            for line in code.split('\n'):
                line = line.strip()
                match = re.match(import_pattern, line)
                if match:
                    if match.group(1):

                        packages = [p.strip() for p in match.group(1).split(',')]
                        for package in packages:
                            if package:
                                package_name = package.split()[0]
                                imports.add(package_name)
                    elif match.group(2):
                        imports.add(match.group(2))

        return imports

    def install_dependencies(self,
                             packages: set) -> None:
        """
        Install dependency packages
        Args:
            packages: python third packages
        Returns:
            None
        """
        for package in packages:
            try:
                __import__(package)
            except ImportError:
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                    self.installed_packages.add(package)
                except subprocess.CalledProcessError as e:
                    logger.warning(f"Failed to install {package}: {str(e)}")

    def uninstall_dependencies(self) -> None:
        """
        Uninstall dependency packages
        Args:
            -
        Returns:
            None
        """
        try:
            for package in self.installed_packages:
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", package])
                except subprocess.CalledProcessError as e:
                    logger.warning(f"Failed to uninstall {package}: {str(e)}")
            self.installed_packages.clear()
        except Exception as e:
            logger.warning(f"Failed to uninstall dependencies: {repr(e)}")

    def reset(self,
              *,
              seed: int | None = None,
              options: Dict[str, str] | None = None) -> Tuple[AgentInput, dict[str, Any]]:
        """
        Reset the executor
        Args:
            seed: -
            options: -
        Returns:
            AgentInput, dict[str, Any]: -
        """
        self.close()
        self.local_namespace = {}
        self.global_namespace = {}
        self.step_finished = True
        self.installed_packages.clear()

    def finished(self) -> bool:
        """
        Check if the executor is finished
        Returns:
            bool: True if finished, False otherwise
        """
        return self.step_finished

    def close(self) -> None:
        """
        Close the executor
        Returns:
            None
        """
        try:
            self.uninstall_dependencies()
            sys.stdout = self.original_stdout
            self.output_buffer.close()
            self.local_namespace.clear()
            self.global_namespace.clear()
        except:
            pass
        finally:
            self.step_finished = True

    def step(
            self,
            actions: List[ActionModel],
            **kwargs) -> Tuple[Observation, float, bool, bool, dict[str, Any]]:
        """
        Step the executor
        Args:
            actions: actions
            **kwargs: -
        Returns:
            Observation, float, bool, bool, dict[str, Any]: -
        """
        self.step_finished = False
        reward = 0
        fail_error = ""
        observation: 'Observation' = Observation(**{
            'dom_tree': '',
            'image': '',
            'action_result': [],
            'info': {}
        })
        try:
            if not actions:
                return (observation, reward,
                        kwargs.get("terminated",
                                   False), kwargs.get("truncated", False), {
                            "exception": "actions is empty"
                        })
            for action in actions:
                code = action.params.get("code", "")
                if not code:
                    continue
                _, output, error = self.execute(code)
                observation.content = output
                observation.action_result.append(
                    ActionResult(is_done=True,
                                 success=False if error else True,
                                 content=f"{output}",
                                 error=f"{error}",
                                 keep=False))
            reward = 1
        except Exception as e:
            fail_error = str(e)
        finally:
            self.step_finished = True

        return (observation, reward, kwargs.get("terminated", False),
                kwargs.get("truncated", False), {
                    "exception": fail_error
                })

    def execute(self, code):
        """
        Execute the code
        Args:
            code: python code
        Returns:
            result, output, error
        """
        required_packages = self.extract_imports(code)
        self.install_dependencies(required_packages)
        sys.stdout = self.output_buffer
        result = None
        error = ''
        output = None
        try:
            # First execute the entire code block
            compiled_code = compile(code, '<string>', 'exec')
            exec(compiled_code, self.global_namespace, self.local_namespace)

            # Get the value of the last line expression
            last_line = code.strip().split('\n')[-1].strip()
            if last_line and not last_line.startswith(
                    ('def ', 'class ', 'if ', 'for ', 'while ')):
                try:
                    # Use eval to get the value of the expression
                    result = eval(last_line, self.global_namespace,
                                  self.local_namespace)
                except Exception as e:
                    error += f'{repr(e)}'
                    logger.warning(f"Error while executing code: {error}")
        except Exception as e:
            error += f'{repr(e)}'
            logger.warning(f"Error while executing code: {error}")
        finally:
            output, error_ = self.get_execute_result()
            error += error_
            self.uninstall_dependencies()

        return result, output, error

    def get_execute_result(self):
        """
        Get the execute result
        Returns:
            output, error
        """
        output = None
        error = ''
        try:
            output = self.output_buffer.getvalue()
            self.output_buffer.truncate(0)
            self.output_buffer.seek(0)
            sys.stdout = self.original_stdout
        except Exception as e:
            error = f'{repr(e)}'
            logger.warning(f"Failed to get output, {repr(e)}")
        return output, error
