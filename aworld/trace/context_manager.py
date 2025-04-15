from typing import Union, Optional, Any, Type, Sequence, Callable
from aworld.trace.trace import Span, Tracer, NoOpSpan
from aworld.trace.auto_trace import AutoTraceModule

class TraceManager:

    def _create_auto_span(self,
                          name: str,
                          attributes: dict[str, Union[str,bool,int,float,Sequence[str],Sequence[bool],Sequence[int],Sequence[float],]] = None
    ) -> Span:

        try:
            # TODO: implement auto tracing
            pass
        except Exception:  # pragma: no cover
            return NoOpSpan()  # ty


    def auto_tracing(self,
                     modules: Union[Sequence[str], Callable[[AutoTraceModule], bool]],
                     min_duration: float
    ) -> None:
        """
        Automatically trace the execution of a function.
        """
        if isinstance(modules, Sequence):
            modules = modules_func_from_sequence(modules)

        

class ContextSpan:
    """A context manager that wraps an existing `Span` object.
    This class provides a way to use a `Span` object as a context manager.
    When the context manager is entered, it returns the `Span` itself.
    When the context manager is exited, it calls `end` on the `Span`.
    Args:
        span: The `Span` object to wrap.
    """

    def __init__(self,
                 span_name: str,
                 tracer: Tracer,
                 attributes: dict[str, Union[str,bool,int,float,Sequence[str],Sequence[bool],Sequence[int],Sequence[float],]] = None) -> None:
        self._span_name = span_name
        self._tracer = tracer
        self._attributes = attributes
        self._span = None

    def _start(self):
        if self._span is not None:
            return

        self._span = self._tracer.start_span(
            name=self._span_name,
            attributes=self._attributes,
        )

    def __enter__(self) -> "Span":
        self._start()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        traceback: Optional[Any],
    ) -> None:
        """Ends context manager and calls `end` on the `Span`."""
        if self._span and self._span.is_recording() and isinstance(exc_val, BaseException):
            self._span.record_exception(exc_val, escaped=True)
        self._span.end()