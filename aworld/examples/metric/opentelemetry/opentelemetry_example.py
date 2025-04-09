import random
import time
from aworld.core.metrics.metric import set_metric_provider,MetricType
from aworld.core.metrics.opentelemetry.opentelemetry_adapter import OpentelemetryMetricProvider
from aworld.core.metrics.context_manager import MetricContext, track_api_metrics
from aworld.core.metrics.template import MetricTemplate

set_metric_provider(OpentelemetryMetricProvider())

my_counter = MetricTemplate(
    type=MetricType.COUNTER,
    name="my_counter",
    description="My custom counter",
    unit="1"
)

my_gauge = MetricTemplate(
    type=MetricType.GAUGE,
    name="my_gauge"
)

my_histogram = MetricTemplate(
    type=MetricType.HISTOGRAM,
    name="my_histogram",
    buckets=[2,4,6,8,10]
)

@track_api_metrics()
def test_api():
    time.sleep(random.uniform(0, 1))


while 1:
    MetricContext.count(my_counter, random.randint(1, 10))
    MetricContext.gauge_set(my_gauge, random.randint(1, 10))
    MetricContext.histogram_record(my_histogram, random.randint(1, 10))
    test_api()
    time.sleep(random.random())