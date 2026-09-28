"""Checkpointing: conversation state in Postgres, memory when not configured."""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import Connection
from psycopg.rows import dict_row

from api.core.settings import get_settings


def get_checkpointer():
    """Postgres checkpointer, so a conversation survives a restart.

    Set DATABASE_URL to enable it. Without it we fall back to the in memory
    saver, which only lives as long as the process.
    """
    database_url = get_settings().database_url
    if not database_url:
        return InMemorySaver()

    connection = Connection.connect(
        database_url,
        autocommit=True,
        prepare_threshold=0,
        row_factory=dict_row,
    )
    checkpointer = PostgresSaver(connection)
    checkpointer.setup()
    return checkpointer
