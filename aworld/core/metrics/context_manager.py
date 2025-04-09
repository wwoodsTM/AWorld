import time
import asyncio
from typing import Callable
from functools import wraps
from aworld.core.metrics.metric import get_metric_provider, MetricType, BaseMetric
from aworld.core.metrics.template import MetricTemplate, MetricTemplates

_GLOBAL_METIRCS={}

class MetricContext:

    @staticmethod
    def get_or_create_metric(template: MetricTemplate):
        if template.name in _GLOBAL_METIRCS:
            return _GLOBAL_METIRCS[template.name]
        
        metric = None
        if template.type == MetricType.COUNTER:
            metric = get_metric_provider().create_counter(template.name, template.description, template.unit, template.labels)
        elif template.type == MetricType.UnDownCounter:
            metric = get_metric_provider().create_updowncounter(template.name, template.description, template.unit, template.labels)
        elif template.type == MetricType.GAUGE:
            metric = get_metric_provider().create_gauge(template.name, template.description, template.unit, template.labels)
        elif template.type == MetricType.HISTOGRAM:
            metric = get_metric_provider().create_histogram(template.name, template.description, template.unit, template.buckets, template.labels)
        
        _GLOBAL_METIRCS[template.name] = metric
        return metric

    @classmethod
    def _validate_type(cls, metric: BaseMetric, type: str):
        if type != metric._type:
            raise ValueError(f"metric type {metric._type} is not {type}")

    @classmethod
    def count(cls, template: MetricTemplate, value: int, labels: dict = None):
        """
        Increment a counter metric.
        """
        metric = cls.get_or_create_metric(template)
        cls._validate_type(metric, MetricType.COUNTER)
        metric.add(value, labels)

    @classmethod
    def inc(cls, template: MetricTemplate, value: int, labels: dict = None):
        """
        Increment a updowncounter metric.
        """
        metric = cls.get_or_create_metric(template)
        cls._validate_type(metric, MetricType.UnDownCounter)
        metric.inc(value, labels)
    
    @classmethod
    def dec(cls, template: MetricTemplate, value: int, labels: dict = None):
        """
        Decrement a updowncounter metric.
        """
        metric = cls.get_or_create_metric(template)
        cls._validate_type(metric, MetricType.UnDownCounter)
        metric.dec(value, labels)

    @classmethod
    def gauge_set(cls, template: MetricTemplate, value: int, labels: dict = None):
        """
        Set a value to a gauge metric.
        """
        metric = cls.get_or_create_metric(template)
        cls._validate_type(metric, MetricType.GAUGE)
        metric.set(value, labels)

    @classmethod
    def histogram_record(cls, template: MetricTemplate, value: int, labels: dict = None):
        """
        Set a value to a histogram metric.
        """
        metric = cls.get_or_create_metric(template)
        cls._validate_type(metric, MetricType.HISTOGRAM)
        metric.record(value, labels)
    

def track_api_metrics():
    """
    Decorator to track API metrics.
    """

    def _record_metrics(func: Callable, start_time: float, status: str) -> None:
        """
        Record metrics for the API.
        """
        method = func.__name__
        elapsed_time = time.time() - start_time
        MetricContext.count(MetricTemplates.REQUEST_COUNT, 1, 
                      labels={"method": method, "status": status})
        MetricContext.histogram_record(MetricTemplates.REQUEST_LATENCY, elapsed_time,
                                 labels={"method": method, "status": status})

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                _record_metrics(func, start_time, "success")
                return result
            except Exception as e:
                _record_metrics(func, start_time, "failure")
                raise e       
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                _record_metrics(func, start_time, "success")
                return result
            except Exception as e:
                _record_metrics(func, start_time, "failure")
                raise e
    
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper

    return decorator