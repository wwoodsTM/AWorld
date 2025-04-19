#!/usr/bin/env python
"""
MCP Server Base Class

This module provides a base class for all MCP server implementations.
It defines a common interface and shared functionality that all MCP
servers should implement.
"""
from abc import ABC, abstractmethod
from typing import Callable, List


def mcp(func):
    """Decorator to mark a method as an MCP function"""
    func.mcp = True
    return func


class MCPServerBase(ABC):
    """
    Base class for all MCP server implementations.

    This class defines the common interface and shared functionality
    that all MCP servers should implement.
    """

    _instance = None

    @classmethod
    def get_instance(cls, *args, **kwargs):
        """
        Get the singleton instance of the server.

        Args:
            *args: Arguments to pass to the constructor if creating a new instance.
            **kwargs: Keyword arguments to pass to the constructor if creating a new instance.

        Returns:
            The singleton instance of the server.
        """
        if cls._instance is None:
            cls._instance = cls(*args, **kwargs)
        return cls._instance

    @classmethod
    def get_functions(cls) -> List[Callable]:
        """
        Get all public methods of the class that can be exposed as MCP functions.

        Returns:
            List of callable methods that can be exposed as MCP functions.
        """
        functions = []
        for name, method in vars(cls).items():
            if hasattr(method, "mcp") and method.mcp == True:
                # Access the method through the class to get the bound method
                bound_method = getattr(cls, name)
                functions.append(bound_method)
        return functions

    def cleanup(self) -> None:
        """
        Clean up resources used by the server.
        This method should be overridden by subclasses if they need to perform cleanup.
        """
        pass

    @abstractmethod
    def _init_server(self) -> None:
        """
        Initialize the server.
        This method should be implemented by all subclasses.
        """
        pass
