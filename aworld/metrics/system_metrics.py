
try:
    from opentelemetry.instrumentation.system_metrics import (
        _DEFAULT_CONFIG,  # type: ignore
        SystemMetricsInstrumentor,
    )
except ImportError:
    raise ImportError(
        "Could not import opentelemetry.instrumentation.system_metrics, please install it with `pip install opentelemetry-instrumentation-system-metrics`"
    )