from logging import Logger, Filter, LogRecord
from abc import ABC, abstractmethod
from typing import Optional
from aworld.trace.trace import get_tracer_provider_silent

class LoggerProvider(ABC):

    @abstractmethod
    def instrumentLogging(self, logger: Logger) -> None:
        """
        Instrument the logger with the backend provider.
        """


_GLOBAL_LOG_PROVIDER: Optional[LoggerProvider] = None

def set_log_provider(provider: str = "otlp",
                     backend: str = "logfire",
                     base_url: str = None,
                     write_token: str = None,
                     **kwargs
):
    """
    Set the global log provider.
    """

    global _GLOBAL_LOG_PROVIDER

    if provider == "otlp":
        from .opentelemetry.otlp_log import OTLPLoggerProvider
        _GLOBAL_LOG_PROVIDER = OTLPLoggerProvider(backend=backend,
                                                  base_url=base_url,
                                                  write_token=write_token,
                                                  **kwargs)



def get_log_provider() -> LoggerProvider:
    """
    Get the global log provider.
    """
    global _GLOBAL_LOG_PROVIDER
    if _GLOBAL_LOG_PROVIDER is None:
        raise ValueError("No log provider has been set.")
    return _GLOBAL_LOG_PROVIDER

class TraceLoggingFilter(Filter):
    """
    A filter that adds trace information to log records.
    """

    def filter(self, record: LogRecord) -> bool:
        """
        Add trace information to the log record.
        """
        trace = get_tracer_provider_silent()
        if trace:
            span = trace.get_current_span()
            record.trace_id = span.get_trace_id() if span else None
            record.span_id = span.get_span_id() if span else None
        return True
