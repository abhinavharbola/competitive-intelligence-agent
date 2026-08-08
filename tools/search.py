import time
import logfire
from tavily import TavilyClient
import config

_client = TavilyClient(api_key=config.TAVILY_API_KEY)


def web_search(query: str, max_results: int = 5) -> str:
    print(f"    [tool] search: {query!r}...", flush=True)
    start = time.time()
    response = _client.search(query=query, max_results=max_results)
    results = response.get("results", [])
    elapsed = time.time() - start
    print(f"    [tool] search returned {len(results)} results in {elapsed:.1f}s", flush=True)
    logfire.info("tool_call", tool="search", query=query, elapsed_seconds=elapsed, result_count=len(results))

    chunks = [f"[{r['title']}]({r['url']})\n{r['content']}" for r in results]
    body = "\n\n".join(chunks) if chunks else "no results"
    return f"<untrusted_web_content>\n{body}\n</untrusted_web_content>"
