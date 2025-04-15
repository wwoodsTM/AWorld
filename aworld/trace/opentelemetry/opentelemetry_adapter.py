from re import S
import sys
import traceback
import time
from threading import Lock
from typing import Union, Optional, Sequence, Any, TYPE_CHECKING, Iterator
from contextvars import Token
from weakref import WeakKeyDictionary, WeakSet
import opentelemetry.context as otlp_context_api
from opentelemetry.sdk.trace import (
    ReadableSpan,
    SpanProcessor,
    Tracer as SDKTracer,
    Span as SDKSpan,
    TracerProvider as SDKTracerProvider,
    SpanKind
)
from opentelemetry.context import Context as OTLPContext
from opentelemetry.semconv.trace import SpanAttributes
from opentelemetry.util import types as otel_types
from aworld.trace.trace import NoOpTracer, SpanType, TraceProvider, Tracer, Span
from .constants import ATTRIBUTES_MESSAGE_KEY
from typing import Optional
from typing import TYPE_CHECKING

def convert_to_otlp_attributes(attr: dict[str, Union[str,bool,int,float,Sequence[str],Sequence[bool],Sequence[int],Sequence[float],]]):
    """Converts a dictionary of attributes to a dictionary of OTLP attributes.
    Args:
        attr: The dictionary of attributes to convert.
    Returns:
        A dictionary of OTLP attributes.
    """

    attributes: dict[str, otel_types.AttributeValue] = {}

    for key, value in attr.items():
        if isinstance(value, (str, bool, int, float)):
            attributes[key] = value
        elif isinstance(value, (list, tuple)):
            if not value:
                attributes[key] = list(value)
            else:
                first_type = type(value[0])
                if all(isinstance(item, first_type) for item in value):
                    attributes[key] = list(value)
                else:
                    raise ValueError(f"Mixed types in list for key {key}")
        else:
            raise ValueError(f"Unsupported type: {type(value)}")

    return attributes


class OTLPTraceProvider(TraceProvider):
    """A TraceProvider that wraps an existing `SDKTracerProvider`.
    This class provides a way to use a `SDKTracerProvider` as a `TraceProvider`.
    When the context manager is entered, it returns the `SDKTracerProvider` itself.
    When the context manager is exited, it calls `shutdown` on the `SDKTracerProvider`.
    Args:
        provider: The internal provider to wrap.
    """ 

    def __init__(self, provider: SDKTracerProvider, suppressed_scopes: Optional[set[str]] = None):
        self._provider: SDKTracerProvider = provider
        self._suppressed_scopes = set()
        if suppressed_scopes:
            self._suppressed_scopes.update(suppressed_scopes)
        self._lock: Lock = Lock()
    
    def get_tracer(
        self,
        name: str,
        version: Optional[str] = None
    ):
        with self._lock:
            if name in self._suppressed_scopes:
                return NoOpTracer()
            else:
                tracer = self._provider.get_tracer(instrumenting_module_name=name, instrumenting_library_version=version)
                return OTLPTracer(tracer)
            
    def shutdown(self) -> None:
        with self._lock:
            if isinstance(self._provider, SDKTracerProvider):
                self._provider.shutdown()

    def force_flush(self, timeout: Optional[float] = None) -> bool:
        with self._lock:
            if isinstance(self._provider, SDKTracerProvider):
                return self._provider.force_flush(timeout)
            else:
                return False

class OTLPTracer(Tracer):
    """A Tracer represents a collection of Spans.
    Args:
        tracer: The internal tracer to wrap.
    """

    def __init__(self, tracer: SDKTracer):
        self._tracer = tracer

    def start_span(
        self,
        name: str,
        span_type: SpanType = SpanType.INTERNAL,
        attributes: dict[str, Union[str,bool,int,float,Sequence[str],Sequence[bool],Sequence[int],Sequence[float],]] = None,
        start_time: Optional[int] = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
    ) -> "Span":
        start_time = start_time or time.time_ns()
        attributes = {**(attributes or {})}
        attributes.setdefault(ATTRIBUTES_MESSAGE_KEY, name)
        otlp_attributes = convert_to_otlp_attributes(attributes)

        span_kind = self._convert_to_span_kind(span_type) if span_type else SpanKind.INTERNAL
        span = self._tracer.start_span(name=name,
                                      kind=span_kind,
                                      attributes=otlp_attributes,
                                      start_time=start_time,
                                      record_exception=record_exception,
                                      set_status_on_exception=set_status_on_exception)
        return OTLPSpan(span)

    def start_as_current_span(
        self,
        name: str,
        span_type: SpanType = SpanType.INTERNAL,
        attributes: dict[str, Union[str,bool,int,float,Sequence[str],Sequence[bool],Sequence[int],Sequence[float],]] = None,
        start_time: Optional[int] = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
        end_on_exit: bool = True,
    ) -> Iterator["Span"]:

        start_time = start_time or time.time_ns()
        attributes = {**(attributes or {})}
        attributes.setdefault(ATTRIBUTES_MESSAGE_KEY, name)
        otlp_attributes = convert_to_otlp_attributes(attributes)
        span_kind = self._convert_to_span_kind(span_type) if span_type else SpanKind.INTERNAL

        with self._tracer.start_as_current_span(name=name,
                                                kind=span_kind,
                                                attributes=otlp_attributes,
                                                start_time=start_time,
                                                record_exception=record_exception,
                                                set_status_on_exception=set_status_on_exception,
                                                end_on_exit=end_on_exit) as span:
            yield OTLPSpan(span)


    def _convert_to_span_kind(self, span_type: SpanType) -> str:
        if span_type == SpanType.INTERNAL:
            return SpanKind.INTERNAL
        elif span_type == SpanType.CLIENT:
            return SpanKind.CLIENT
        elif span_type == SpanType.SERVER:
            return  SpanKind.SERVER
        elif span_type == SpanType.PRODUCER:
            return SpanKind.PRODUCER
        elif span_type == SpanType.CONSUMER:
            return SpanKind.CONSUMER
        else:
            return SpanKind.INTERNAL


class OTLPSpan(Span, ReadableSpan):
    """A Span represents a single operation within a trace.
    """

    def __init__(self, span: SDKSpan):
        self._span = span
        OPEN_SPANS.add(self)
        self._token: Optional[Token[OTLPContext]] = None
        self._attach()

    if not TYPE_CHECKING:  # pragma: no branch
        def __getattr__(self, name: str) -> Any:
            return getattr(self._span, name)

    def end(self, end_time: Optional[int] = None) -> None:
        OPEN_SPANS.discard(self)
        end_time = end_time or time.time_ns()
        self._span.end(end_time=end_time)
        self._detach()

    def set_attribute(self, key: str, value: Any) -> None:
        self._span.set_attribute(key=key, value=value)

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        self._span.set_attributes(attributes=attributes)

    def is_recording(self) -> bool:
        return self._span.is_recording()

    def record_exception(
            self,
            exception: BaseException,
            attributes: dict[str, Any] = None,
            timestamp: Optional[int] = None,
            escaped: bool = False,
        ) -> None:
        timestamp = timestamp or time.time_ns()
        attributes = {**(attributes or {})}

        otlp_attributes = convert_to_otlp_attributes(attributes)
        if exception is not sys.exc_info()[1]:
            # OTEL's record_exception uses `traceback.format_exc()` which is for the current exception,
            # ignoring the passed exception.
            # So we override the stacktrace attribute with the correct one.
            stacktrace = ''.join(traceback.format_exception(type(exception), exception, exception.__traceback__))
            otlp_attributes[SpanAttributes.EXCEPTION_STACKTRACE] = stacktrace
        
        self._span.record_exception(exception=exception,
                                   attributes=otlp_attributes,
                                   timestamp=timestamp,
                                   escaped=escaped)

    def _attach(self):
        if self._token is not None:
            return
        self._token = otlp_context_api.attach(otlp_context_api.set_span_in_context(self))

    def _detach(self):
        if self._token is None:
            return
        otlp_context_api.detach(self._token)
        self._token = None


OPEN_SPANS: WeakSet[OTLPSpan] = WeakSet()