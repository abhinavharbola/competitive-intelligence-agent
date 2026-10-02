import time
import logfire
from tavily import TavilyClient
import config
from tools.results import NO_RESULTS, wrap_untrusted

_client = TavilyClient(api_key=config.TAVILY_API_KEY)

_RETRY_ATTEMPTS = 2
_RETRY_DELAY_SECONDS = 2


def _search_with_retry(query: str, max_results: int) -> dict:
    last_error = None
    for attempt in range(1, _RETRY_ATTEMPTS + 1):
        try:
            return _client.search(query=query, max_results=max_results)
        except Exception as e:
            last_error = e
            logfire.warn("tool_call_failed", tool="search", query=query, attempt=attempt, error=str(e))
            if attempt < _RETRY_ATTEMPTS:
                print(f"    [tool] search attempt {attempt} failed ({e}), retrying", flush=True)
                time.sleep(_RETRY_DELAY_SECONDS)
    raise last_error


def web_search(query: str, max_results: int = 5) -> str:
    print(f"    [tool] search: {query!r}...", flush=True)
    start = time.time()
    response = _search_with_retry(query, max_results)
    results = response.get("results", [])
    elapsed = time.time() - start
    print(f"    [tool] search returned {len(results)} results in {elapsed:.1f}s", flush=True)
    logfire.info("tool_call", tool="search", query=query, elapsed_seconds=elapsed, result_count=len(results))

    chunks = [f"[{r['title']}]({r['url']})\n{r['content']}" for r in results]
    body = "\n\n".join(chunks) if chunks else NO_RESULTS
    return wrap_untrusted(body)
