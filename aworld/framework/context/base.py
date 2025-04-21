# coding: utf-8
# Copyright (c) 2025 inclusionAI.
from aworld.config import ConfigDict
from aworld.framework.context.session import Session
from aworld.framework.singleton import InheritanceSingleton


class Context(InheritanceSingleton):
    user: str
    session_id: str
    session: Session = None
    task_id: str

    ext_info: ConfigDict


# global context
context = Context()
