import requests
from opentelemetry.exporter.otlp.proto.http import Compression
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import SynchronousMultiSpanProcessor, TracerProvider as SDKTracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from aworld.trace.trace import set_tracer_provider
from aworld.trace.opentelemetry.opentelemetry_adapter import OTLPTraceProvider
from aworld.trace.context_manager import TraceManager


logfire_api_url = "https://logfire-us.pydantic.dev/v1/traces" 
logfire_token = ""
headers = {'User-Agent': f'logfire/3.14.0', 'Authorization': logfire_token}
session = requests.Session()
session.headers.update(headers)
span_exporter = OTLPSpanExporter(
                        endpoint=logfire_api_url,
                        session=session,
                        compression=Compression.Gzip,
                    )

processor = SynchronousMultiSpanProcessor()
processor.add_span_processor(SimpleSpanProcessor(span_exporter))
set_tracer_provider(OTLPTraceProvider(SDKTracerProvider(active_span_processor=processor)))
trace = TraceManager()

with trace.span("hello") as span:
    span.set_attribute("hello", "aworld")
    print("hello aworld")


