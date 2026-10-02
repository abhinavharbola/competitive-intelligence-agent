import json
import threading
import time
import logfire
from openai import OpenAI
from google import genai
import config

_LLM_TIMEOUT_SECONDS = 60
_RETRY_ATTEMPTS = 3
_RETRY_BASE_DELAY = 2
_RATE_LIMIT_BASE_DELAY = 10

_nim = OpenAI(api_key=config.NIM_API_KEY, base_url=config.NIM_BASE_URL, timeout=_LLM_TIMEOUT_SECONDS, max_retries=0)
_groq_executor = OpenAI(api_key=config.GROQ_EXECUTOR_API_KEY, base_url=config.GROQ_BASE_URL, timeout=_LLM_TIMEOUT_SECONDS, max_retries=0)
_groq_judge = OpenAI(api_key=config.GROQ_JUDGE_API_KEY or "unset", base_url=config.GROQ_BASE_URL, timeout=_LLM_TIMEOUT_SECONDS, max_retries=0)
_gemini = genai.Client(api_key=config.GEMINI_API_KEY)

_NON_RETRYABLE_STATUS = (400, 401, 403, 404, 422)
_DAILY_QUOTA_MARKERS = ("perday", "per day", "daily")
_NON_RETRYABLE_MARKERS = (
    "invalid_api_key", "invalid api key", "incorrect api key",
    "api key not valid", "api_key_invalid",
    "unauthorized", "permission_denied", "permission denied",
    "authentication",
)


def _status_of(e: Exception) -> int | None:
    for attr in ("status_code", "code"):
        value = getattr(e, attr, None)
        if isinstance(value, int):
            return value
    return None


def _is_retryable(e: Exception) -> bool:
    status = _status_of(e)
    text = str(e).lower()
    if status in _NON_RETRYABLE_STATUS:
        return False
    if status == 429 and any(marker in text for marker in _DAILY_QUOTA_MARKERS):
        return False
    return not any(marker in text for marker in _NON_RETRYABLE_MARKERS)


def _with_retries(fn, label: str = "llm"):
    last_error = None
    for attempt in range(1, _RETRY_ATTEMPTS + 1):
        try:
            return fn()
        except Exception as e:
            last_error = e
            logfire.warn("llm_call_failed", call=label, attempt=attempt, error=str(e))
            if not _is_retryable(e):
                raise
            if attempt < _RETRY_ATTEMPTS:
                base = _RATE_LIMIT_BASE_DELAY if _status_of(e) == 429 else _RETRY_BASE_DELAY
                delay = base * (2 ** (attempt - 1))
                print(f"  [llm] attempt {attempt} failed ({e}), retrying in {delay}s...", flush=True)
                time.sleep(delay)
    raise last_error


def _run_with_timeout(fn, seconds: float = _LLM_TIMEOUT_SECONDS):
    box: dict = {}

    def target():
        try:
            box["value"] = fn()
        except BaseException as e:
            box["error"] = e

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(seconds)
    if thread.is_alive():
        raise TimeoutError(f"call exceeded {seconds}s")
    if "error" in box:
        raise box["error"]
    return box["value"]


def _call_openai_compatible(client: OpenAI, model: str, system: str, user: str) -> dict:
    print(f"  [llm] calling {model}...", flush=True)
    start = time.time()

    def attempt():
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
        )
        return response, json.loads(response.choices[0].message.content)

    response, parsed = _with_retries(attempt, label=model)
    elapsed = time.time() - start
    print(f"  [llm] {model} responded in {elapsed:.1f}s", flush=True)
    usage = getattr(response, "usage", None)
    logfire.info(
        "llm_call",
        model=model,
        elapsed_seconds=elapsed,
        prompt_tokens=usage.prompt_tokens if usage else None,
        completion_tokens=usage.completion_tokens if usage else None,
    )
    return parsed


def call_planner(system: str, user: str) -> dict:
    return _call_openai_compatible(_nim, config.PLANNER_MODEL, system, user)


def call_executor(system: str, user: str) -> dict:
    return _call_openai_compatible(_groq_executor, config.EXECUTOR_MODEL, system, user)


def call_judge(system: str, user: str) -> dict:
    if not config.GROQ_JUDGE_API_KEY:
        raise RuntimeError("GROQ_JUDGE_API_KEY not set, required for eval/judge.py")
    return _call_openai_compatible(_groq_judge, config.JUDGE_MODEL, system, user)


def call_gemini(model: str, system: str, user: str) -> dict:
    print(f"  [llm] calling {model}...", flush=True)
    start = time.time()

    def attempt():
        response = _run_with_timeout(lambda: _gemini.models.generate_content(
            model=model,
            contents=user,
            config={"system_instruction": system, "response_mime_type": "application/json"},
        ))
        return response, json.loads(response.text)

    response, parsed = _with_retries(attempt, label=model)
    elapsed = time.time() - start
    print(f"  [llm] {model} responded in {elapsed:.1f}s", flush=True)
    usage = response.usage_metadata
    logfire.info(
        "llm_call",
        model=model,
        elapsed_seconds=elapsed,
        prompt_tokens=usage.prompt_token_count if usage else None,
        completion_tokens=usage.candidates_token_count if usage else None,
    )
    return parsed
