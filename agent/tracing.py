import time
import logfire
import config

logfire.configure(token=config.LOGFIRE_TOKEN or None, send_to_logfire=bool(config.LOGFIRE_TOKEN))


def traced_node(name):
    def decorator(fn):
        def wrapper(state):
            print(f"[node] {name} starting...", flush=True)
            start = time.time()
            with logfire.span("node", node=name):
                result = fn(state)
            elapsed = time.time() - start
            print(f"[node] {name} finished in {elapsed:.1f}s", flush=True)
            logfire.info("node_finished", node=name, elapsed_seconds=elapsed)
            return result
        return wrapper
    return decorator
