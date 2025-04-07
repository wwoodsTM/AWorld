import random
import time
from aworld.core.metrics.metric import set_metric_provider,get_metric_provider
from aworld.core.metrics.prometheus.prometheus_adapter import PrometheusConsoleMetricExporter, PrometheusMetricProvider

set_metric_provider(PrometheusMetricProvider(PrometheusConsoleMetricExporter(out_interval_secs=2)))
my_counter = get_metric_provider().create_counter("my_counter", "test_counter_desc", "count")

while 1:
    my_counter.add(random.randint(1, 10), )
    time.sleep(random.random())