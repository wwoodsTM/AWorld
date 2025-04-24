from math import log
import requests
import time
from logging import NOTSET, Logger, Handler, LogRecord, Formatter
from typing import Union
from urllib.parse import urljoin
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http import Compression
from opentelemetry.sdk._logs import LoggerProvider as SDKLoggerProvider
from opentelemetry.sdk._logs._internal import SynchronousMultiLogRecordProcessor
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from aworld.trace.trace import get_tracer_provider_silent, AttributeValueType, Tracer

from ..log import LoggerProvider, TraceLoggingFilter

TRACE_LOG_FORMAT = '%(asctime)s - [%(trace_id)s] - [%(span_id)s] - %(name)s - %(levelname)s - %(message)s'

class OTLPLoggerProvider(LoggerProvider):

    def __init__(self,
                 backend: str = None,
                 base_url: str = None,
                 write_token: str = None,
                 **kwargs
    ) -> None:
        self._logger_provider = _build_otlp_privider(backend=backend,
                                               base_url=base_url,
                                               write_token=write_token)

    def instrumentLogging(self, logger: Logger, level: Union[int, str] = NOTSET) -> None:
        """
        Instrument the logger with the otlp provider.
        """
        for handler in logger.root.handlers:
            handler.setFormatter(Formatter(TRACE_LOG_FORMAT))
            handler.addFilter(TraceLoggingFilter())

        logger.addHandler(OPLPLogginHandler(level))


class OPLPLogginHandler(Handler):
    """
    A handler class which writes logging records, appropriately formatted,
    to a stream. Note that this class does not close the stream, as
    sys.stdout or sys.stderr may be used.
    """
    def __init__(self,
                 level: Union[int, str] = NOTSET,
                 tracer_name: str = "aworld.log",
    ) -> None:
        """
        Initialize the handler.
        """
        super().__init__(level=level)
        self._tracer_name = tracer_name
        # self.streamHandler = StreamHandler()
        self._tracer: Tracer = None

    def emit(self, record: LogRecord) -> None:
        """
        Emit a record.
        """
        trace = get_tracer_provider_silent()
        if not trace or not trace.get_current_span() or not trace.get_current_span().is_recording():
            return

        if not self._tracer:
            self._tracer = trace.get_tracer(name=self._tracer_name)

        try:
            msg_template = record.msg
            attributes = {
                'code.filepath': record.pathname,
                'code.lineno': record.lineno,
                'code.function': record.funcName,
                'log.level': record.levelname,
                'log.logger': record.name,
                'log.message': self.format(record),
            }
            self._create_span(
                span_name=msg_template,
                attributes=attributes,
                exc_info=record.exc_info,
            )
        except RecursionError:  # See issue 36272
            raise
        except Exception:
            self.handleError(record)

    def _create_span(self,
                     span_name: str,
                     attributes: dict[str, AttributeValueType] = None,
                     exc_info: BaseException = None):
        start_time = time.time_ns()
        span = self._tracer.start_span(
            name=span_name,
            attributes=attributes,
            start_time=start_time,
        )
        if exc_info:
            span.record_exception(exception=exc_info, timestamp=start_time)
        span.end()


def _build_otlp_privider(backend: str = None, base_url: str = None, write_token: str = None):
    """Build the otlp provider.
    Args:
        backends: The backends to use.
        write_token: The write token to use.
        **kwargs: Additional keyword arguments to pass to the provider.
    Returns:
        The otlp provider.
    """
    backend = backend or "logfire"
    if backend == "logfire":
        log_exporter = _configure_logfire_exporter(write_token=write_token, base_url=base_url)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    otlp_log_processor = BatchLogRecordProcessor(log_exporter)
    multi_log_processor = SynchronousMultiLogRecordProcessor()
    multi_log_processor.add_log_record_processor(otlp_log_processor)
    logger_provider = SDKLoggerProvider()
    logger_provider.add_log_record_processor(multi_log_processor)
    return logger_provider


def _configure_logfire_exporter(write_token: str, base_url: str = None) -> None:
    """Configure the Logfire exporter.
    Args:
        write_token: The write token to use.
        base_url: The base URL to use.
        **kwargs: Additional keyword arguments to pass to the exporter.
    """
    base_url = base_url or "https://logfire-us.pydantic.dev"
    headers = {'User-Agent': f'logfire/3.14.0', 'Authorization': write_token}
    session = requests.Session()
    session.headers.update(headers)
    return OTLPLogExporter(
                endpoint=urljoin(base_url, '/v1/logs'),
                session=session,
                compression=Compression.Gzip,
    )
