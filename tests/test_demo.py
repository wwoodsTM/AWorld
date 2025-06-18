import unittest
from unittest.mock import patch, MagicMock
import os


from aworld.core.tool.base import ToolFactory


class TestShellTool(unittest.TestCase):


    def test_init(self):
        """Test initialization"""
        self.assertEqual(self.shell_tool.type, "function")

