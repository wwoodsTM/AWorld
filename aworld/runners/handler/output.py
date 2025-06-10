# aworld/runners/handler/output.py
import json
from aworld.core.task import TaskResponse
from aworld.models.model_response import ModelResponse
from aworld.runners.handler.base import DefaultHandler
from aworld.output.base import StepOutput, MessageOutput, ToolResultOutput, Output
from aworld.core.common import TaskItem
from aworld.core.context.base import Context
from aworld.core.event.base import Message, Constants
from aworld.logs.util import logger
from aworld.runners.utils import TaskType

class DefaultOutputHandler(DefaultHandler):
    def __init__(self, runner):
        self.runner = runner

    async def handle(self, message):
        if message.category != Constants.OUTPUT:
            return
        # 1. get outputs
        outputs = self.runner.task.outputs
        if not outputs:
            yield Message(
                category=Constants.TASK,
                payload=TaskItem(msg="Cannot get outputs.", data=message, stop=True),
                sender=self.name(),
                session_id=Context.instance().session_id,
                topic=TaskType.ERROR
            )
            return
        # 2. build Output
        payload = message.payload
        try:
            if isinstance(payload, Output):
                await outputs.add_output(payload)
            elif isinstance(payload, TaskResponse):
                await outputs.add_output(
                    Output(
                        data=f"usage: {json.dumps(payload.usage)}"
                    )
                )
                if message.topic == TaskType.FINISHED or message.topic == TaskType.ERROR:
                    await outputs.mark_completed()
            else:
                output = MessageOutput(source=payload)
                await outputs.add_output(output)
        except Exception as e:
            logger.warning(f"Failed to parse output: {e}")
            yield Message(
                category=Constants.TASK,
                payload=TaskItem(msg="Failed to parse output.", data=payload, stop=True),
                sender=self.name(),
                session_id=Context.instance().session_id,
                topic=TaskType.ERROR
            )

        return