# coding: utf-8
# Copyright (c) 2025 inclusionAI.
import abc
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from aworld.core.context.base import Context


class Runner(object):
    __metaclass__ = abc.ABCMeta

    _use_demon: bool = False
    daemon_target: Callable[..., Any] = None
    context: 'Context' = None

    async def pre_run(self):
        pass

    async def post_run(self):
        pass

    @abc.abstractmethod
    async def do_run(self, context: 'Context' = None):
        """Raise exception if not success."""

    async def _daemon_run(self):
        if self._use_demon and self.daemon_target and callable(self.daemon_target):
            import threading
            t = threading.Thread(target=self.daemon_target, name="daemon", daemon=True)
            t.start()

    async def run(self) -> Any:
        try:
            await self.pre_run()
            await self._daemon_run()
            ret = await self.do_run(self.context)
            return 0 if ret is None else ret
        except BaseException as ex:
            self._exception = ex
            # do record or report
            raise ex
        finally:
            await self.post_run()
