from typing import AsyncGenerator, Generator, Callable, Any

from aworld.logs.util import logger
from aworld.output import OutputChannel
from aworld.output.base import OutputPart, MessageOutput


async def consume_channel_messages(channel: OutputChannel, callback: Callable[..., Any]):
    """Consume messages from the message panel"""
    message_panel = channel.message_renderer.panel

    try:
        async for message in message_panel.get_messages_async():
            logger.info(f"Consumer: Found message: {message}")
            if isinstance(message, MessageOutput):
                ## parts
                if message.parts:
                    await consume(message.parts, callback)
                ## parts
                elif message.reason_generator or message.response_generator:
                    if message.reason_generator:
                        await consume(message.reason_generator, callback)
                    if message.reason_generator:
                        await consume(message.response_generator, callback)
                else:
                    await consume(message.reasoning, callback)
                    await consume(message.response, callback)

    except Exception as e:
        logger.error(f"Error during message consumption: {e}")


async def consume(message_item, callback: Callable[..., Any]):
    if not message_item:
        return
    if isinstance(message_item, AsyncGenerator):
        async for part in message_item:
            if isinstance(part, OutputPart):
                callback(part)
            else:
                callback(message_item)

    elif isinstance(message_item, Generator) or isinstance(message_item, list):
        for part in message_item:
            if isinstance(part, OutputPart):
                callback(part)
            else:
                callback(message_item)
    elif isinstance(message_item, str):
        callback(message_item)
