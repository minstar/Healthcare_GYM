import logging
import os
from typing import Optional


logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger().setLevel(logging.ERROR)


def _get_relative_path(pathname: str) -> str:
    path_parts = pathname.split(os.sep)
    if "agent_environment" in path_parts:
        agent_env_index = path_parts.index("agent_environment")
        return os.sep.join(path_parts[agent_env_index:])
    return pathname


class RelativePathFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        if hasattr(record, "pathname"):
            record.pathname = _get_relative_path(record.pathname)
        return super().format(record)


def create_logger(name: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    simple_formatter = RelativePathFormatter(
        "%(asctime)s %(name)s [%(pathname)s:%(lineno)d] - %(levelname)s:%(message)s"
    )
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    return logger
