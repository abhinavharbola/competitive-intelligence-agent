OPEN_TAG = "<untrusted_web_content>"
CLOSE_TAG = "</untrusted_web_content>"
NO_RESULTS = "no results"


def wrap_untrusted(body: str) -> str:
    return f"{OPEN_TAG}\n{body}\n{CLOSE_TAG}"


EMPTY_RESULT = wrap_untrusted(NO_RESULTS)


def is_empty_result(result: str) -> bool:
    return result.strip() == EMPTY_RESULT
