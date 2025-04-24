# coding: utf-8
# Copyright (c) 2025 inclusionAI.
import os
from aworld.metrics.context_manager import MetricContext
from aworld.logs.util import logger

if os.getenv("LOGFIRE_WRITE_TOKEN"):
    MetricContext.configure(provider="otlp",
                            backend="logfire",
                            write_token=os.getenv("LOGFIRE_WRITE_TOKEN")
    )
else:
    logger.warning("No LOGFIRE_WRITE_TOKEN found. Using console as backend.")
    MetricContext.configure(provider="otlp",
                            backend="console")
