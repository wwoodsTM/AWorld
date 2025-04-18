import aworld.trace as trace
from examples.trace.autotrace_demo import TestClassB


def main():
    trace.trace_configure(
        backends=["console"],
        write_token="Your logfire write token, create guide refer to https://logfire.pydantic.dev/docs/how-to-guides/create-write-tokens/"
    )

    trace.auto_tracing("examples.trace.*", 0.01)

    with trace.span("hello") as span:
        span.set_attribute("parent_test_attr", "pppppp")
        print("hello aworld")
        with trace.span("child hello") as span2:
            span2.set_attribute("child_test_attr", "cccccc")
            print("child hello aworld")
            current_span = trace.get_current_span()
            print("trace_id=", current_span.get_trace_id())

    b = TestClassB()
    b.classb_function_1()
    b.classb_function_2()
    b.classb_function_1()
    b.classb_function_2()


if __name__ == "__main__":
    main()
