import asyncio
import time
from typing import AsyncGenerator, Callable, Awaitable
from pydantic import BaseModel, Field
import sys
import traceback
import logging
import os
import uuid
logger = logging.getLogger(__name__)


class Pipe:
    class Valves(BaseModel):
        GAIA_MODEL_ID: str = Field(
            default="claude-3-7-sonnet,gpt_4o",
            description="Gaia模型ID，多模型名可使用`,`分隔",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.data_prefix = "data:"
        self.emitter = None

    def pipes(self):
        models = self.valves.GAIA_MODEL_ID.split(",")

        return [
            {
                "id": model.strip(),
                "name": f"gaia_agent@{model.strip()}",
            }
            for model in models
        ]

    async def pipe(
        self, body: dict, __event_emitter__: Callable[[dict], Awaitable[None]] = None
    ) -> AsyncGenerator[str, None]:

        self.emitter = __event_emitter__
        process = None

        try:
            logger.info(f">>> Gaia Agent: body={body}")

            prompt = body["messages"][-1]["content"]
            model = body["model"]
            logger.info(f">>> Gaia Agent: prompt={prompt}, model={model}")

            # 准备命令
            cmd = [
                sys.executable,
                "-m",
                "examples.gaia.gaia_agent_runner",
                "--provider",
                "openai",
                "--model",
                model,
                "--prompt",
                prompt,
            ]

            # cmd = ["ping", "-c", "5", "www.baidu.com"]

            # 创建并启动子进程
            process = await asyncio.subprocess.create_subprocess_exec(
                *cmd,
                cwd="/app/aworld",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                stdin=asyncio.subprocess.DEVNULL,
                env=os.environ.copy(),
            )

            logger.info(f">>> Gaia Agent: process={process}")

            while True:
                line = await process.stdout.readline()
                if not line:
                    break

                line = line.decode("utf-8").rstrip()
                logger.info(f">>> Gaia Agent: line={line}")
                yield self._wrap_line(f"{line}\n")

            # 等待进程结束
            return_code = await process.wait()
            yield self._wrap_line(f"Process exited with code {return_code}\n")
            await asyncio.sleep(0.1)

        except Exception as e:
            emsg = traceback.format_exc()
            logger.error(f">>> Gaia Agent: exception {emsg}")
            yield self._wrap_line(f"Gaia Agent Error: {emsg}\n")

        finally:
            # 确保进程被终止
            if process:
                try:
                    logger.info("Stopping gaia agent process...")
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=3.0)
                        logger.info("Stopping gaia agent process timeout!")
                    except asyncio.TimeoutError:
                        logger.warning(
                            "Stopping gaia agent process timeout, force kill!"
                        )
                        process.kill()
                        await process.wait()
                        logger.info("Gaia agent process force killed!")
                except Exception as e:
                    logger.error(f"Error stopping gaia agent process: {e}")
            yield self._wrap_line(f"Gaia Task End!")

    def _wrap_line(self, line: str) -> str:
        line = line.replace("<think>", "<_think_>")
        line = line.replace("</think>", "<_think_/>")
        return line
