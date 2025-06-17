# coding: utf-8
# Copyright (c) 2025 inclusionAI.
from collections import Counter, OrderedDict
from typing import Dict

from aworld.config import ConfigDict
from aworld.core.context.session import Session
from aworld.core.singleton import InheritanceSingleton
from aworld.utils.common import nest_dict_counter


class Context(InheritanceSingleton):
    """Single instance, can use construction or `instance` static method to create or get `Context` instance.

    Examples:
        >>> context = Context()
    or
        >>> context = Context.instance()
    """

    def __init__(self,
                 user: str = None,
                 *,
                 task_id: str = None,
                 trace_id: str = None,
                 session: Session = None,
                 engine: str = None):
        self._user = user
        self._init(task_id=task_id, trace_id=trace_id, session=session, engine=engine)

    def _init(self, *, task_id: str = None, trace_id: str = None, session: Session = None, engine: str = None):
        self._task_id = task_id
        self._engine = engine
        self._trace_id = trace_id
        self._session: Session = session
        self.context_info = ConfigDict()
        self.agent_info = ConfigDict()
        self.trajectories = OrderedDict()
        self._token_usage = {
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "total_tokens": 0
        }

        self._event_manager = None

    def add_token(self, usage: Dict[str, int]):
        self._token_usage = nest_dict_counter(self._token_usage, usage)

    def reset(self, **kwargs):
        self._init(**kwargs)

    @property
    def trace_id(self):
        return self._trace_id

    @trace_id.setter
    def trace_id(self, trace_id):
        self._trace_id = trace_id

    @property
    def token_usage(self):
        return self._token_usage

    @property
    def engine(self):
        return self._engine

    @engine.setter
    def engine(self, engine: str):
        self._engine = engine

    @property
    def user(self):
        return self._user

    @user.setter
    def user(self, user):
        if user is not None:
            self._user = user

    @property
    def task_id(self):
        return self._task_id

    @task_id.setter
    def task_id(self, task_id):
        if task_id is not None:
            self._task_id = task_id

    @property
    def session_id(self):
        if self.session:
            return self.session.session_id
        else:
            return None

    @property
    def session(self):
        return self._session

    @session.setter
    def session(self, session: Session):
        self._session = session

    @property
    def event_manager(self):
        return self._event_manager

    @event_manager.setter
    def event_manager(self, event_manager: 'EventManager'):
        self._event_manager = event_manager

    @property
    def record_path(self):
        return "."

    @property
    def is_task(self):
        return True

    @property
    def enable_visible(self):
        return False

    @property
    def enable_failover(self):
        return False

    @property
    def enable_cluster(self):
        return False
