import time
import threading
from typing import Sequence, Optional
from prometheus_client import Counter, Gauge, Histogram, start_http_server, generate_latest, REGISTRY
from aworld.core.metrics.metric import BaseMetricProvider, BaseCounter, BaseUpDownCounter, BaseMetricExporter, BaseGauge, BaseHistogram


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
    
    def create_counter(self, name: str, description: str, unit: str, labelnames: Optional[Sequence[str]] = None) -> BaseCounter:
        """
        Create a counter metric.
        Args:
            name: The name of the metric.
            description: The description of the metric.
            unit: The unit of the metric.
        Returns:
            The counter metric.
        """
        return PrometheusCounter(name, description, unit, self, labelnames)
    
    def create_un_down_counter(self, name: str, description: str, unit: str, labelnames: Optional[Sequence[str]] = None) -> BaseUpDownCounter:
        """
        Create an up-down counter metric.
        Args:
            name: The name of the metric.
            description: The description of the metric.
            unit: The unit of the metric.
        Returns:
            The up-down counter metric.
        """
        return PrometheusUpDownCounter(name, description, unit, self, labelnames)

    def create_gauge(self, name: str, description: str, unit: str, labelnames: Optional[Sequence[str]] = None) -> BaseGauge:
        """
        Create a gauge metric.
        Args:
            name: The name of the metric.
            description: The description of the metric.
            unit: The unit of the metric.
        Returns:
            The gauge metric.
        """
        return PrometheusGauge(name, description, unit, self, labelnames)

    def create_histogram(self,
                         name: str, 
                         description: str, 
                         unit: str, 
                         buckets: Optional[Sequence[float]] = None, 
                         labelnames: Optional[Sequence[str]] = None) -> BaseHistogram:
        """
        Create a histogram metric.
        Args:
            name: The name of the metric.
            description: The description of the metric.
            unit: The unit of the metric.
            buckets: The buckets of the histogram.
        Returns:
            The histogram metric.
        """
        return PrometheusHistogram(name, description, unit, self, buckets, labelnames)

class PrometheusCounter(BaseCounter):
    """
    PrometheusCounter is a subclass of BaseCounter, representing a counter metric for Prometheus.
    """
    def __init__(self, 
                 name: str, 
                 description: str, 
                 unit: str, 
                 provider: BaseMetricProvider, 
                 labelnames: Optional[Sequence[str]] = None):
        """
        Initialize the PrometheusCounter.
        Args:
            name: The name of the metric.
            description: The description of the metric.
            unit: The unit of the metric.
            provider: The provider of the metric.
        """
        labelnames = labelnames or []
        super().__init__(name, description, unit, provider, labelnames)
        self._counter = Counter(name=name, documentation=description, labelnames=labelnames, unit=unit)
    
    def add(self, value: int, labels: dict = None) -> None:
        """
        Add a value to the counter.
        Args:
            value: The value to add to the counter.
            labels: The labels to associate with the value.
        """
        if labels:
            self._counter.labels(**labels).inc(value)
        else:
            self._counter.inc(value)

class PrometheusUpDownCounter(BaseUpDownCounter):
    """
    PrometheusUpDownCounter is a subclass of BaseUpDownCounter, representing an up-down counter metric for Prometheus.
    """
    def __init__(self, 
                 name: str, 
                 description: str, 
                 unit: str, 
                 provider: BaseMetricProvider,
                 labelnames: Optional[Sequence[str]] = None):
        """
        Initialize the PrometheusUpDownCounter.
        Args:
            name: The name of the metric.
            description: The description of the metric.
            unit: The unit of the metric.
            provider: The provider of the metric.
        """
        labelnames = labelnames or []
        super().__init__(name, description, unit, provider, labelnames)
        self._gauge = Gauge(name=name, documentation=description, labelnames=labelnames, unit=unit)

    def inc(self, value: int, labels: dict = None) -> None:
        """
        Add a value to the counter.
        Args:
            value: The value to add to the counter.
            labels: The labels to associate with the value.
        """
        if labels:
            self._gauge.labels(**labels).inc(value)
        else:
            self._gauge.inc(value)
    
    def dec(self, value: int, labels: dict = None) -> None:
        """
        Subtract a value from the counter.
        Args:
            value: The value to subtract from the counter.
            labels: The labels to associate with the value.
        """
        if labels:
            self._gauge.labels(**labels).dec(value)
        else:
            self._gauge.dec(value)
        

class PrometheusGauge(BaseGauge):
    """
    PrometheusGauge is a subclass of BaseGauge, representing a gauge metric for Prometheus.
    """
    def __init__(self,
                 name: str, 
                 description: str, 
                 unit: str, 
                 provider: BaseMetricProvider,
                 labelnames: Optional[Sequence[str]] = None):
        """
        Initialize the PrometheusGauge.
        Args:
            name: The name of the metric.
            description: The description of the metric.
            unit: The unit of the metric.
            provider: The provider of the metric.
        """
        labelnames = labelnames or []
        super().__init__(name, description, unit, provider, labelnames)
        self._gauge = Gauge(name=name, documentation=description, labelnames=labelnames, unit=unit)
    
    def set(self, value: int, labels: dict = None) -> None:
        """
        Set the value of the gauge.
        Args:
            value: The value to set the gauge to.
            labels: The labels to associate with the value.
        """
        if labels:
            self._gauge.labels(**labels).set(value)
        else:
            self._gauge.set(value)
    
    def inc(self, value: int, labels: dict = None) -> None:
        """
        Add a value to the gauge.
        Args:
            value: The value to add to the gauge.
            labels: The labels to associate with the value.
        """
        if labels:
            self._gauge.labels(**labels).inc(value) 
        else:
            self._gauge.inc(value)

    def dec(self, value: int, labels: dict = None) -> None:
        """
        Subtract a value from the gauge.
        Args:
            value: The value to subtract from the gauge.
            labels: The labels to associate with the value.
        """
        if labels:
            self._gauge.labels(**labels).dec(value)
        else:
            self._gauge.dec(value)

class PrometheusHistogram(BaseHistogram):
    """
    PrometheusHistogram is a subclass of BaseHistogram, representing a histogram metric for Prometheus.
    """
    def __init__(self, 
                 name: str, 
                 description: str, 
                 unit: str, 
                 provider: BaseMetricProvider, 
                 buckets: Sequence[float] = None,
                 labelnames: Optional[Sequence[str]] = None):
        """
        Initialize the PrometheusHistogram.
        Args:
            name: The name of the metric.
            description: The description of the metric.
            unit: The unit of the metric.
            provider: The provider of the metric.
        """
        labelnames = labelnames or []
        super().__init__(name, description, unit, provider, buckets, labelnames)
        if buckets:
            self._histogram = Histogram(name=name, documentation=description, labelnames=labelnames, unit=unit, buckets=buckets)
        else:
            self._histogram = Histogram(name=name, documentation=description, labelnames=labelnames, unit=unit)
    
    def record(self, value: int, labels: dict = None) -> None:
        """
        Record a value in the histogram.
        Args:
            value: The value to record in the histogram.
            labels: The labels to associate with the value.
        """
        if labels:
            self._histogram.labels(**labels).observe(value)
        else:
            self._histogram.observe(value)

class PrometheusMetricExporter(BaseMetricExporter):
    """
    PrometheusMetricExporter is a class for exporting metrics to Prometheus.
    """
    def __init__(self, provider: BaseMetricProvider, port: int=8000):
        """
        Initialize the PrometheusMetricExporter.
        Args:
            provider: The provider of the metrics.
        """
        super().__init__(provider)
        self.port = port
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
            print("==============================================")
            print(metrics_text.decode('utf-8'))
            time.sleep(self.out_interval_secs)

    def shutdown(self) -> None:
        """
        Shutdown the PrometheusConsoleMetricExporter.
        """
        self._should_shutdown = True

