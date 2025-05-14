"""Add config table

Revision ID: ca81bd47c050
Revises: 7e5b5dc7342b
Create Date: 2024-08-25 15:26:35.241684

"""

from typing import Sequence, Union

from fastapi import params
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "ca81bd47c050"
down_revision: Union[str, None] = "7e5b5dc7342b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.create_table(
        "config",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("data", sa.JSON(), nullable=False),
        sa.Column("version", sa.Integer, nullable=False),
        sa.Column(
            "created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            nullable=True,
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
        ),
    )
    init_gaia_config()

def downgrade():
    op.drop_table("config")

def init_gaia_config():
    import json
    prompt_suggestions = None
    with(open("/app/aworld/examples/gaia/GAIA/2023/validation/metadata.jsonl", 'r', 'utf-8')) as f:
        data_set = [json.loads(line) for line in f]

        prompts = [
            {"title": [i["Question"], i["task_id"]], "content": {"task_id": i["task_id"], "Question": i["Question"]}}
            for i in data_set
        ]
        prompt_suggestions = json.dumps(prompts, ensure_ascii=False, indent=None)

    op.execute(
        f"INSERT INTO config (id, data, version) VALUES (1, %s, 0)",
        params=(prompt_suggestions)
    )
    print(f">>> patch gaia_agent: add prompt_suggestions success!")

    func_content = None
    with(open("/app/aworld/examples/gaia/openwebui-patch/gaia_agent.py", 'r', 'utf-8')) as f:
        func_content = f.read()
    op.execute(
        f"INSERT INTO function (id, user_id, name, type, content, meta, is_activate, is_global) VALUES('gaia_agent', '00000000-0000-0000-0000-000000000000', 'pipe', %s, %s,1,1)",
        params={func_content, '{"description": "gaia_agent", "manifest": {}}'},
    )
    print(f">>> patch gaia_agent: add gaia_agent function success!")
