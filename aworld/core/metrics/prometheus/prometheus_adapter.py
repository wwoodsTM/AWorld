import time
import threading
from prometheus_client import Counter, start_http_server, generate_latest, REGISTRY
from aworld.metrics.metric import BaseMetric, BaseMetricProvider, BaseCounter, BaseMetricExporter


class PrometheusMetricProvider(BaseMetricProvider):
    """
    PrometheusMetricProvider is a subclass of BaseMetricProvider, representing a metric provider for Prometheus.
    """

    def __init__(self, exporter: BaseMetricExporter):
        """
        Initialize the PrometheusMetricProvider.
        Args:
            port: The port to use for the Prometheus server.
        """
        super().__init__()
        self.exporter = exporter
    
    def shutdown(self) -> None:
        """
        Shutdown the PrometheusMetricProvider.
        """
        self.exporter.shutdown()
    
    def create_counter(self, name: str, description: str, unit: str) -> BaseCounter:
        """
        Create a counter metric.
        Args:
            name: The name of the metric.
            description: The description of the metric.
            unit: The unit of the metric.
        Returns:
            The counter metric.
        """
        return PrometheusCounter(name, description, unit, self)

class PrometheusCounter(BaseCounter):
    """
    PrometheusCounter is a subclass of BaseCounter, representing a counter metric for Prometheus.
    """
    def __init__(self, name: str, description: str, unit: str, provider: BaseMetricProvider):
        """
        Initialize the PrometheusCounter.
        Args:
            name: The name of the metric.
            description: The description of the metric.
            unit: The unit of the metric.
            provider: The provider of the metric.
        """
        super().__init__(name, description, unit, provider)
        self._counter = Counter(name=name, documentation=description, labelnames=[], unit=unit)
    
    def add(self, value: int, labels: dict = None) -> None:
        """
        Add a value to the counter.
        Args:
            value: The value to add to the counter.
            labels: The labels to associate with the value.
        """
        labels = labels or {}
        self._counter.labels(**labels).inc(value)

class PrometheusMetricExporter(BaseMetricExporter):
    """
    PrometheusMetricExporter is a class for exporting metrics to Prometheus.
    """
    def __init__(self, provider: BaseMetricProvider):
        """
        Initialize the PrometheusMetricExporter.
        Args:
            provider: The provider of the metrics.
        """
        super().__init__(provider)
        server, server_thread = start_http_server(self.port)
        self.server = server
        self.server_thread = server_thread

    def shutdown(self) -> None:
        """
        Shutdown the PrometheusMetricExporter.
        """
        self.server.shutdown()
        self.server_thread.join()
    
class PrometheusConsoleMetricExporter(BaseMetricExporter):
    """Implementation of :class:`BaseMetricExporter` that prints metrics to the
    console.

    This class can be used for diagnostic purposes. It prints the exported
    metrics to the console STDOUT.
    """

    def __init__(self, out_interval_secs: float=1.0):
        """Initialize the console exporter."""
        self._should_shutdown = False
        self.out_interval_secs = out_interval_secs
        # Bug修复: 调用当前类的方法时需要加上 `self.`
        self.metrics_thread = threading.Thread(target=self._output_metrics_to_console)
        self.metrics_thread.start()

    def _output_metrics_to_console(self):
        while not self._should_shutdown:
            metrics_text = generate_latest(REGISTRY)
            print(metrics_text.decode('utf-8'))
            time.sleep(self.out_interval_secs)

    def shutdown(self) -> None:
        """
        Shutdown the PrometheusConsoleMetricExporter.
        """
        self._should_shutdown = True

