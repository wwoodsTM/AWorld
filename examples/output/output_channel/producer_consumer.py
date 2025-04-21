import asyncio
from typing import List, AsyncGenerator, Generator

from aworld.core.task import TaskModel
from aworld.output import MessageOutput, WorkSpace, Artifact, ArtifactType
from aworld.output.base import OutputPart
from aworld.output.observer import on_artifact_create
from aworld.output.output_channel import OutputChannel
from aworld.logs.util import logger


@on_artifact_create(workspace_id="workspace-producer-consumer", 
                   filters={"artifact_type": "CODE"})
async def handle_code_artifacts(artifact, **kwargs):
    """Handle only code artifacts"""
    logger.info(f"Code consumer received artifact in workspace {kwargs['workspace_id']}")
    logger.info(f"Code content: {artifact.content}")

class Producer:
    def __init__(self, task_id: str, workspace: WorkSpace):
        self.channel = OutputChannel.create(task_id=task_id, workspace=workspace)
        self.task = TaskModel(name=f"task_{task_id}", outputs=self.channel)

    async def produce_message_parts(self, parts: List[OutputPart]):
        """Produce a message output"""
        message = MessageOutput(parts=parts)
        await asyncio.sleep(0.1)
        await self.task.outputs.add_output(message)
        logger.info("Producer: Created message output")

    async def produce_message(self, source: str):
        """Produce a message output"""
        message = MessageOutput(source=source)
        await asyncio.sleep(0.1)
        await self.task.outputs.add_output(message)
        logger.info("Producer: Created message output")

    async def mark_message_complete(self):
        """Mark the message panel as completed"""
        await self.task.outputs.message_renderer.panel.mark_completed()

    async def produce_artifact(self, artifact_type: ArtifactType, content: str, metadata: dict):
        """Produce an artifact output"""
        artifact = Artifact(
            artifact_type=artifact_type,
            content=content,
            metadata=metadata
        )
        await asyncio.sleep(0.3)
        await self.task.outputs.add_output(artifact)
        logger.info(f"Producer: Created {artifact_type} artifact")

class Consumer:
    def __init__(self, channel: OutputChannel):
        self.channel = channel

    async def consume_messages(self):
        """Consume messages from the message panel"""
        logger.info("Starting message consumption")
        message_panel = self.channel.message_renderer.panel
        
        try:
            async for message in message_panel.get_messages_async():
                logger.info(f"Consumer: Found message: {message}")
                if isinstance(message, MessageOutput):
                    ## parts
                    if message.parts:
                        await self.log_message(message.parts)
                    ## parts
                    elif message.reason_generator or message.response_generator:
                        if message.reason_generator:
                            await self.log_message(message.reason_generator)
                        if message.reason_generator:
                            await self.log_message(message.response_generator)
                    else:
                        await self.log_message(message.reasoning)
                        await self.log_message(message.response)

        except Exception as e:
            logger.error(f"Error during message consumption: {e}")

    async def log_message(self, item):
        if not item:
            return
        if isinstance(item, AsyncGenerator):
            async for part in item:
                if isinstance(part, OutputPart):
                    logger.info(f"- Part content: {part.content}, metadata: {part.metadata}")
                else:
                    logger.info(f"- Part content: {item}")

        elif isinstance(item, Generator) or isinstance(item,list):
            for part in item:
                if isinstance(part, OutputPart):
                    logger.info(f"- Part content: {part.content}, metadata: {part.metadata}")
                else:
                    logger.info(f"- Part content: {item}")
        elif isinstance(item, str):
            logger.info(f"- Part content: {item}")


async def run():
    # Create workspace
    workspace = WorkSpace.from_local_storages(workspace_id="workspace-producer-consumer")
    
    # Create producer and consumer
    producer = Producer(task_id="producer-001", workspace=workspace)
    consumer = Consumer(producer.channel)

    # Start consumer task
    consumer_task = asyncio.create_task(consumer.consume_messages())
    
    # Produce messages
    await producer.produce_message(source="Message === START =====")

    await producer.produce_message_parts([
        OutputPart(content="Second message part1", metadata={"type": "text"}),
        OutputPart(content="Second message part2", metadata={"type": "text"}),
        OutputPart(content="Second message part3", metadata={"type": "text"})
    ])

    await producer.produce_message_parts([
        OutputPart(content="Second message part1", metadata={"type": "text"}),
        OutputPart(content="Second message part2", metadata={"type": "text"}),
        OutputPart(content="Second message part3", metadata={"type": "text"})
    ])

    # Mark messages as complete
    await producer.mark_message_complete()

    # Add an artifact
    await producer.produce_artifact(
        artifact_type=ArtifactType.CODE,
        content="print('Hello World')",
        metadata={"language": "python"}
    )

    # Wait for consumer to finish
    await consumer_task

if __name__ == '__main__':
    asyncio.run(run()) 