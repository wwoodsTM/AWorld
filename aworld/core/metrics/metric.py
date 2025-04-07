from abc import ABC, abstractmethod

_global_metric_provider = None

def set_metric_provider(provider):
    """
    Set the global metric provider.
    """
    global _global_metric_provider
    _global_metric_provider = provider

def get_metric_provider():
    """
    Get the global metric provider.
    """
    global _global_metric_provider
    if _global_metric_provider is None:
        raise ValueError("No metric provider has been set.")
    return _global_metric_provider

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
        pass

    def add_exporter(self, exporter):
        """
        Add an exporter to the list of exporters.
        """
        self._exporters.append(exporter)

    @abstractmethod
    def create_counter(self, name: str, description: str, unit) -> "BaseCounter":
        """
        Create a counter.

        Args:
            name: The name of the instrument to be created
            description: A description for this instrument and what it measures.
            unit: The unit for observations this instrument reports. For
                example, ``By`` for bytes. UCUM units are recommended.
        """
        pass  


class BaseMetric(ABC):
    """
    Metric is the base class for all metrics.
    Args:
        name: The name of the metric.
        description: The description of the metric.
        unit: The unit of the metric.
        provider: The provider of the metric.
    """
    def __init__(self, name: str, description: str, unit: str, provider: BaseMetricProvider):
        self._name = name
        self._description = description
        self._unit = unit
        self.provider = provider

    @abstractmethod
    def record_metric(self, value: int, labels: dict = None) -> None:
        """
        Record a metric.
        Args:
            value: The value to record.
            labels: The labels to associate with the value.
        """
        pass

class BaseCounter(BaseMetric):
    """
    BaseCounter is a subclass of BaseMetric, representing a counter metric.
    A counter is a cumulative metric that represents a single numerical value that only ever goes up.
    """

    def record_metric(self, value: int, labels: dict = None) -> None:
        """
        Record a metric.
        Args:
            value: The value to record.
            labels: The labels to associate with the value.
        """
        self.add(value, labels)

    @abstractmethod
    def add(self, value: int, labels: dict = None) -> None:
        """
        Add a value to the counter.
        Args:
            value: The value to add to the counter.
            labels: The labels to associate with the value.
        """
        pass

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
        pass