# Create a task with output channel
import asyncio
import logging

from aworld.core.task import TaskModel
from aworld.output import MessageOutput, WorkSpace, Artifact, ArtifactType
from aworld.output.base import OutputPart
from aworld.output.observer import on_artifact_create
from aworld.output.output_channel import OutputChannel

@on_artifact_create(workspace_id="workspace-task-001", filters={"artifact_type": "CODE"})
async def handle_specific_artifacts(artifact, **kwargs):
    logging.info(f"{artifact.artifact_type} artifact created in specific workspace {kwargs['workspace_id']}: {artifact.artifact_id}")

async def run():
    workspace = WorkSpace.from_local_storages(workspace_id="workspace-task-001")

    channel = OutputChannel.create(
        task_id="task-001",
        workspace=workspace
    )

    task = TaskModel(
        name="task_1",
        outputs=channel
    )

    # Add a message output with parts
    message = MessageOutput(
        parts=[
            OutputPart(content="Part 1", metadata={"type": "text"}),
            OutputPart(content="Part 2", metadata={"type": "code"})
        ]
    )
    await task.outputs.add_output(message)
    

    # Add an artifact
    code_artifact = Artifact(
        artifact_type=ArtifactType.CODE,
        content="print('Hello World')",
        metadata={"language": "python"}
    )
    await task.outputs.add_output(code_artifact)



if __name__ == '__main__':
    asyncio.run(run())
