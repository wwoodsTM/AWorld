import traceback

from loguru._logger import Logger

from aworld.logs.util import logger


def customize_logger(agent: str, method: str) -> Logger:
    """
    Customize the logger to log messages with different colors and levels.
    """
    logger_context = logger.bind(agent=agent, method=method)
    try:
        # define customized log function
        def log_plan(message):
            logger_context.log("PLAN", message, color="<cyan>")

        def log_execute(message):
            logger_context.log("EXECUTE", message, color="<blue>")

        logger_context.add(
            "agent_details.log", format="{time} {level} {message}", level="INFO"
        )

        logger_context.level("PLAN", no=25, color="<bold><magenta>")
        logger_context.level("EXECUTE", no=25, color="<bold><yellow>")

        logger_context.plan = log_plan
        logger_context.execute = log_execute
    except Exception as e:
        logger_context.warning(traceback.format_exc())
    return logger_context


if __name__ == "__main__":
    log_context = customize_logger("agent", "method")
    log_context.plan("This is a plan message.")
    log_context.execute("This is a execute message.")
