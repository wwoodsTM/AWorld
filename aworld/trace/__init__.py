# coding: utf-8
# Copyright (c) 2025 inclusionAI.
from pickle import GLOBAL

from aworld.trace.context_manager import TraceManager, trace_configure


GLOBAL_TRACE_MANAGER: TraceManager = TraceManager()
span = GLOBAL_TRACE_MANAGER.span
auto_tracing = GLOBAL_TRACE_MANAGER.auto_tracing
get_current_span = GLOBAL_TRACE_MANAGER.get_current_span
new_manager = GLOBAL_TRACE_MANAGER.get_current_span