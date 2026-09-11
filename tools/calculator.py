import time
import logfire
from simpleeval import simple_eval


def calculate(expression: str) -> str:
    print(f"    [tool] calculator: {expression!r}", flush=True)
    start = time.time()
    try:
        result = str(simple_eval(expression))
    except Exception as e:
        logfire.info("tool_call", tool="calculator", expression=expression, elapsed_seconds=time.time() - start, success=False)
        raise
    logfire.info("tool_call", tool="calculator", expression=expression, elapsed_seconds=time.time() - start, success=True)
    return result