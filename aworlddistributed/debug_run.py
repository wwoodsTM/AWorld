import os

import uvicorn
from dotenv import load_dotenv

from aworld.metrics import MetricContext

if __name__ == "__main__":
    load_dotenv()
    # os.environ["OTLP_TRACES_ENDPOINT"] = "http://localhost:4318/v1/traces"
    MetricContext.configure(provider="otlp",
                            backend="antmonitor"
                            )
    import main
    uvicorn.run(main.app, host="0.0.0.0", port=9999)
