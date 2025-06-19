# coding: utf-8
# Copyright (c) 2025 inclusionAI.
import asyncio
import inspect
import os
import pkgutil
import re
import socket
import sys
import threading
import time

from functools import wraps
from types import FunctionType
from typing import Callable, Any, Tuple, List, Iterator, Dict, Union

from aworld.logs.util import logger


def convert_to_snake(name: str) -> str:
    """Class name convert to snake."""
    if '_' not in name:
        name = re.sub(r'([a-z])([A-Z])', r'\1_\2', name)
    return name.lower()


def snake_to_camel(snake):
    words = snake.split('_')
    return ''.join([w.capitalize() for w in words])


def is_abstract_method(cls, method_name):
    method = getattr(cls, method_name)
    return (hasattr(method, '__isabstractmethod__') and method.__isabstractmethod__) or (
            isinstance(method, FunctionType) and hasattr(
        method, '__abstractmethods__') and method in method.__abstractmethods__)


def override_in_subclass(name: str, sub_cls: object, base_cls: object) -> bool:
    """Judge whether a subclass overrides a specified method.

    Args:
        name: The method name of sub class and base class
        sub_cls: Specify subclasses of the base class.
        base_cls: The parent class of the subclass.

    Returns:
        Overwrite as true in subclasses, vice versa.
    """
    if not issubclass(sub_cls, base_cls):
        logger.warning(f"{sub_cls} is not sub class of {base_cls}")
        return False

    if sub_cls == base_cls and hasattr(sub_cls, name) and not is_abstract_method(sub_cls, name):
        return True

    this_method = getattr(sub_cls, name)
    base_method = getattr(base_cls, name)
    return this_method is not base_method


def _walk_to_root(path: str) -> Iterator[str]:
    """Yield directories starting from the given directory up to the root."""
    if not os.path.exists(path):
        yield ''

    if os.path.isfile(path):
        path = os.path.dirname(path)

    last_dir = None
    current_dir = os.path.abspath(path)
    while last_dir != current_dir:
        yield current_dir
        parent_dir = os.path.abspath(os.path.join(current_dir, os.path.pardir))
        last_dir, current_dir = current_dir, parent_dir


def find_file(filename: str) -> str:
    """Find file from the folders for the given file.

    NOTE: Current running path priority, followed by the execution file path, and finally the aworld package path.

    Args:
        filename: The file name that you want to search.
    """

    def run_dir():
        try:
            main = __import__('__main__', None, None, fromlist=['__file__'])
            return os.path.dirname(main.__file__)
        except ModuleNotFoundError:
            return os.getcwd()

    path = os.getcwd()
    if os.path.exists(os.path.join(path, filename)):
        path = os.getcwd()
    elif os.path.exists(os.path.join(run_dir(), filename)):
        path = run_dir()
    else:
        frame = inspect.currentframe()
        current_file = __file__

        while frame.f_code.co_filename == current_file or not os.path.exists(
                frame.f_code.co_filename
        ):
            assert frame.f_back is not None
            frame = frame.f_back
        frame_filename = frame.f_code.co_filename
        path = os.path.dirname(os.path.abspath(frame_filename))

    for dirname in _walk_to_root(path):
        if not dirname:
            continue
        check_path = os.path.join(dirname, filename)
        if os.path.isfile(check_path):
            return check_path

    return ''


def search_in_module(module: object, base_classes: List[type]) -> List[Tuple[str, type]]:
    """Find all classes that inherit from a specific base class in the module."""
    results = []
    for name, obj in inspect.getmembers(module, inspect.isclass):
        for base_class in base_classes:
            if issubclass(obj, base_class) and obj is not base_class:
                results.append((name, obj))
    return results


def _scan_package(package_name: str, base_classes: List[type], results: List[Tuple[str, type]] = []):
    try:
        package = sys.modules[package_name]
    except:
        return

    try:
        for sub_package, name, is_pkg in pkgutil.walk_packages(package.__path__):
            try:
                __import__(f"{package_name}.{name}")
            except:
                continue

            if is_pkg:
                _scan_package(package_name + "." + name, base_classes, results)
            try:
                module = __import__(f"{package_name}.{name}", fromlist=[name])
                results.extend(search_in_module(module, base_classes))
            except:
                continue
    except:
        pass


def scan_packages(package: str, base_classes: List[type]) -> List[Tuple[str, type]]:
    results = []
    _scan_package(package, base_classes, results)
    return results


class ReturnThread(threading.Thread):
    def __init__(self, func, *args, **kwargs):
        threading.Thread.__init__(self)
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.result = None
        self.daemon = True

    def run(self):
        self.result = asyncio.run(self.func(*self.args, **self.kwargs))


def asyncio_loop():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    return loop


def sync_exec(async_func: Callable[..., Any], *args, **kwargs):
    """Async function to sync execution."""
    if not asyncio.iscoroutinefunction(async_func):
        return async_func(*args, **kwargs)

    loop = asyncio_loop()
    if loop and loop.is_running():
        thread = ReturnThread(async_func, *args, **kwargs)
        thread.start()
        thread.join()
        result = thread.result
    else:
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(async_func(*args, **kwargs))
    return result


def nest_dict_counter(usage: Dict[str, Union[int, Dict[str, int]]],
                      other: Dict[str, Union[int, Dict[str, int]]],
                      ignore_zero: bool = True):
    """Add counts from two dicts or nest dicts."""
    result = {}
    for elem, count in usage.items():
        # nest dict
        if isinstance(count, Dict):
            res = nest_dict_counter(usage[elem], other.get(elem, {}))
            result[elem] = res
            continue

        newcount = count + other.get(elem, 0)
        if not ignore_zero or newcount > 0:
            result[elem] = newcount

    for elem, count in other.items():
        if elem not in usage and not ignore_zero:
            result[elem] = count
    return result


def get_class(module_class: str):
    import importlib

    assert module_class
    module_class = module_class.strip()
    idx = module_class.rfind('.')
    if idx != -1:
        module = importlib.import_module(module_class[0:idx])
        class_names = module_class[idx + 1:].split(":")
        cls_obj = getattr(module, class_names[0])
        for inner_class_name in class_names[1:]:
            cls_obj = getattr(cls_obj, inner_class_name)
        return cls_obj
    else:
        raise Exception("{} can not find!".format(module_class))


def new_instance(module_class: str, *args, **kwargs):
    """Create module class instance based on module name."""
    return get_class(module_class)(*args, **kwargs)


def retryable(tries: int = 3, delay: int = 1):
    def inner_retry(f):
        @wraps(f)
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 0:
                try:
                    return f(*args, **kwargs)
                except Exception as e:
                    msg = f"{str(e)}, Retrying in {mdelay} seconds..."
                    logger.warning(msg)
                    time.sleep(mdelay)
                    mtries -= 1
            return f(*args, **kwargs)

        return f_retry

    return inner_retry


def get_local_ip():
    try:
        # build UDP socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # connect to an external address (no need to connect)
        s.connect(("8.8.8.8", 80))
        # get local IP
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "127.0.0.1"

def replace_env_variables(config) -> Any:
    if isinstance(config, dict):
        for key, value in config.items():
            if isinstance(value, str):
                # 处理字符串中嵌入的环境变量
                if "${" in value and "}" in value:
                    # 寻找所有 ${ENV_VAR} 模式的子串
                    pattern = r'\${([^}]+)}'
                    matches = re.findall(pattern, value)
                    result = value
                    for env_var_name in matches:
                        env_var_value = os.getenv(env_var_name, f"${{{env_var_name}}}")
                        result = result.replace(f"${{{env_var_name}}}", env_var_value)
                    config[key] = result
                    logger.info(f"Replaced {value} with {config[key]}")
            if isinstance(value, dict) or isinstance(value, list):
                replace_env_variables(value)
    elif isinstance(config, list):
        for index, item in enumerate(config):
            if isinstance(item, str):
                # 处理字符串中嵌入的环境变量
                if "${" in item and "}" in item:
                    pattern = r'\${([^}]+)}'
                    matches = re.findall(pattern, item)
                    result = item
                    for env_var_name in matches:
                        env_var_value = os.getenv(env_var_name, f"${{{env_var_name}}}")
                        result = result.replace(f"${{{env_var_name}}}", env_var_value)
                    config[index] = result
                    logger.info(f"Replaced {item} with {config[index]}")
            if isinstance(item, dict) or isinstance(item, list):
                replace_env_variables(item)
    return config
