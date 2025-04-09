from abc import ABC, abstractmethod
from tkinter import NO
from typing import Optional, Sequence

class MetricType:
    """
    MetricType is a class for defining the type of a metric.
    """
    COUNTER = "counter"
    UnDownCounter = "undowncounter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"

class BaseMetricProvider(ABC):
    """
    MeterProvider is the entry point of the API. 
    """

    def __init__(self):
        # list of exporters
        self._exporters = []

    @abstractmethod
    def shutdown(self):
        """
        shutdown the metric provider.
        """

    def add_exporter(self, exporter):
        """
        Add an exporter to the list of exporters.
        """
        self._exporters.append(exporter)

    @abstractmethod
    def create_counter(self, name: str, description: str, unit: str, labelnames: Optional[Sequence[str]] = None) -> "BaseCounter":
        """
        Create a counter.

        Args:
            name: The name of the instrument to be created
            description: A description for this instrument and what it measures.
            unit: The unit for observations this instrument reports. For
                example, ``By`` for bytes. UCUM units are recommended.
        """

    @abstractmethod
    def create_un_down_counter(self, name: str, description: str, unit: str, labelnames: Optional[Sequence[str]] = None) -> "BaseUnDownCounter":
        """
        Create a un-down counter.
        Args:
            name: The name of the instrument to be created
            description: A description for this instrument and what it measures.
            unit: The unit for observations this instrument reports. For
                example, ``By`` for bytes. UCUM units are recommended.
        """ 

    @abstractmethod  
    def create_gauge(self, name: str, description: str, unit: str, labelnames: Optional[Sequence[str]] = None) -> "BaseGauge":
        """
        Create a gauge.
        Args:
            name: The name of the instrument to be created
            description: A description for this instrument and what it measures.
            unit: The unit for observations this instrument reports. For
                example, ``By`` for bytes. UCUM units are recommended.
        """

    @abstractmethod
    def create_histogram(self,
                         name: str, 
                         description: str, 
                         unit: str, 
                         buckets: Optional[Sequence[float]] = None,
                         labelnames: Optional[Sequence[str]] = None) -> "BaseHistogram":
        """
        Create a histogram.
        Args:
            name: The name of the instrument to be created
            description: A description for this instrument and what it measures.
            unit: The unit for observations this instrument reports. For
                example, ``By`` for bytes. UCUM units are recommended.
        """


class BaseMetric(ABC):
    """
    Metric is the base class for all metrics.
    Args:
        name: The name of the metric.
        description: The description of the metric.
        unit: The unit of the metric.
        provider: The provider of the metric.
    """
    def __init__(self, 
                 name: str, 
                 description: str, 
                 unit: str, 
                 provider: BaseMetricProvider, 
                 labelnames: Optional[Sequence[str]] = None):
        self._name = name
        self._description = description
        self._unit = unit
        self._provider = provider
        self._labelnames = labelnames
        self._type = None

class BaseCounter(BaseMetric):
    """
    BaseCounter is a subclass of BaseMetric, representing a counter metric.
    A counter is a cumulative metric that represents a single numerical value that only ever goes up.
    """
    def __init__(self,
                 name: str, 
                 description: str, 
                 unit: str, 
                 provider: BaseMetricProvider, 
                 labelnames: Optional[Sequence[str]] = None):
        """
        Initialize the Counter.
        Args:
            name: The name of the metric.
            description: The description of the metric.
            unit: The unit of the metric.
            provider: The provider of the metric.
        """
        super().__init__(name, description, unit, provider, labelnames)
        self._type = MetricType.COUNTER

    @abstractmethod
    def add(self, value: int, labels: dict = None) -> None:
        """
        Add a value to the counter.
        Args:
            value: The value to add to the counter.
            labels: The labels to associate with the value.
        """

class BaseUpDownCounter(BaseMetric):
    """
    BaseUnDownCounter is a subclass of BaseMetric, representing an un-down counter metric.
    An un-down counter is a cumulative metric that represents a single numerical value that only ever goes up.
    """
    def __init__(self,
                 name: str, 
                 description: str, 
                 unit: str, 
                 provider: BaseMetricProvider, 
                 labelnames: Optional[Sequence[str]] = None):
        """
        Initialize the UnDownCounter.
        Args:
            name: The name of the metric.
            description: The description of the metric.
            unit: The unit of the metric.
            provider: The provider of the metric.
        """
        super().__init__(name, description, unit, provider, labelnames)
        self._type = MetricType.UnDownCounter

    @abstractmethod
    def inc(self, value: int, labels: dict = None) -> None:
        """
        Add a value to the gauge.
        Args:
            value: The value to add to the gauge.
            labels: The labels to associate with the value.
        """

    @abstractmethod
    def dec(self, value: int, labels: dict = None) -> None:
        """
        Subtract a value from the gauge.
        Args:
            value: The value to subtract from the gauge.
            labels: The labels to associate with the value.
        """

class BaseGauge(BaseMetric):
    """
    BaseGauge is a subclass of BaseMetric, representing a gauge metric.
    A gauge is a metric that represents a single numerical value that can arbitrarily go up and down.
    """

    def __init__(self,
                 name: str, 
                 description: str, 
                 unit: str, 
                 provider: BaseMetricProvider, 
                 labelnames: Optional[Sequence[str]] = None):
        """
        Initialize the Gauge.
        Args:
            name: The name of the metric.
            description: The description of the metric.
            unit: The unit of the metric.
            provider: The provider of the metric.
        """
        super().__init__(name, description, unit, provider, labelnames) 
        self._type = MetricType.GAUGE

    @abstractmethod
    def set(self, value: int, labels: dict = None) -> None:
        """
        Set the value of the gauge.
        Args:
            value: The value to set the gauge to.
            labels: The labels to associate with the value.
        """

class BaseHistogram(BaseMetric):
    """
    BaseHistogram is a subclass of BaseMetric, representing a histogram metric.
    A histogram is a metric that represents the distribution of a set of values.
    """
    def __init__(self, 
                 name: str, 
                 description: str, 
                 unit: str,
                 provider: BaseMetricProvider, 
                 buckets: Sequence[float] = None, 
                 labelnames: Optional[Sequence[str]] = None):
        """
        Initialize the Histogram.
        Args:
            name: The name of the metric.
            description: The description of the metric.
            unit: The unit of the metric.
            provider: The provider of the metric.
            buckets: The buckets of the histogram.
        """
        super().__init__(name, description, unit, provider, labelnames)
        self._type = MetricType.HISTOGRAM
        self._buckets = buckets

    @abstractmethod
    def record(self, value: int, labels: dict = None) -> None:
        """
        Record a value in the histogram.
        Args:
            value: The value to record in the histogram.
            labels: The labels to associate with the value.
        """

class BaseMetricExporter(ABC):
    """
    MetricExporter is a class for exporting metrics.
    """
    def __init__(self, provider: BaseMetricProvider):
        """
        Initialize the MetricExporter.
        Args:
            provider: The provider of the metric.
        """
        self.provider = provider

    @abstractmethod
    def shutdown(self):
        """
        Export the metrics.
        """


_GLOBAL_METRIC_PROVIDER: Optional[BaseMetricProvider] = None

def set_metric_provider(provider):
    """
    Set the global metric provider.
    """
    global _GLOBAL_METRIC_PROVIDER
    _GLOBAL_METRIC_PROVIDER = provider

def get_metric_provider():
    """
    Get the global metric provider.
    """
    global _GLOBAL_METRIC_PROVIDER
    if _GLOBAL_METRIC_PROVIDER is None:
        raise ValueError("No metric provider has been set.")
    return _GLOBAL_METRIC_PROVIDER