import random
import time
from aworld.core.metrics.metric import set_metric_provider,get_metric_provider
from aworld.core.metrics.opentelemetry.opentelemetry_adapter import OpentelemetryMetricProvider

set_metric_provider(OpentelemetryMetricProvider())
my_counter = get_metric_provider().create_counter("my_counter", "test_counter_desc", "count")

while 1:
    my_counter.add(random.randint(1, 10), )
    time.sleep(random.random())