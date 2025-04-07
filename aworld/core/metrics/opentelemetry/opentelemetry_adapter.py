from aworld.core.metrics.metric import BaseMetric, BaseMetricProvider, BaseCounter, BaseMetricExporter
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, ConsoleMetricExporter

class OpentelemetryMetricProvider(BaseMetricProvider):
    """
    MetricProvider is a class for providing metrics.
    """

    def __init__(self, exporter: BaseMetricExporter=None):
        """
        Initialize the MetricProvider.
        Args:
            exporter: The exporter of the metric.
        """ 
        super().__init__()
        if not exporter:
            exporter = ConsoleMetricExporter()
        self.exporter = exporter
        self._otel_provider = MeterProvider(metric_readers=[PeriodicExportingMetricReader(exporter=self.exporter, export_interval_millis=5000)])
        metrics.set_meter_provider(self._otel_provider)
        self._meter = self._otel_provider.get_meter("aworld")

    def create_counter(self, name: str, description: str, unit: str) -> BaseCounter:
        """
        Create a counter.
        Args:
            name: The name of the counter.
            description: The description of the counter.
            unit: The unit of the counter.
        """
        return OpentelemetryCounter(name, description, unit, self)
    
    def shutdown(self):
        """
        Shutdown the metric provider.
        """
        pass

class OpentelemetryCounter(BaseCounter):
    """
    Counter is a subclass of BaseCounter, representing a counter metric.
    A counter is a cumulative metric that represents a single numerical value that only ever goes up.
    """
    def __init__(self, name: str, description: str, unit: str, provider: OpentelemetryMetricProvider):
        """
        Initialize the Counter.
        Args:
            name: The name of the counter.
            description: The description of the counter.
            unit: The unit of the counter.
            provider: The provider of the counter.
        """
        super().__init__(name, description, unit, provider)
        self._counter = provider._meter.create_counter(name, description, unit)

    def add(self, value: int, labels: dict = None) -> None:
        """
        Add a value to the counter.
        Args:
            value: The value to add to the counter.
            labels: The labels to associate with the value.
        """
        if labels is None:
            labels = {}
        self._counter.add(value, labels)

