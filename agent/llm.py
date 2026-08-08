import json
import time
import logfire
from openai import OpenAI
from google import genai
import config

_LLM_TIMEOUT_SECONDS = 60
_RETRY_ATTEMPTS = 3
_RETRY_BASE_DELAY = 2

_nim_planner = OpenAI(api_key=config.NIM_PLANNER_API_KEY, base_url=config.NIM_BASE_URL, timeout=_LLM_TIMEOUT_SECONDS, max_retries=1)
_nim_judge = OpenAI(api_key=config.NIM_JUDGE_API_KEY, base_url=config.NIM_BASE_URL, timeout=_LLM_TIMEOUT_SECONDS, max_retries=1)
_groq = OpenAI(api_key=config.GROQ_API_KEY, base_url=config.GROQ_BASE_URL, timeout=_LLM_TIMEOUT_SECONDS, max_retries=1)
_gemini = genai.Client(api_key=config.GEMINI_API_KEY)


def _with_retries(fn):
    last_error = None
    for attempt in range(1, _RETRY_ATTEMPTS + 1):
        try:
            return fn()
        except Exception as e:
            last_error = e
            if attempt < _RETRY_ATTEMPTS:
                delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                print(f"  [llm] attempt {attempt} failed ({e}), retrying in {delay}s...", flush=True)
                time.sleep(delay)
    raise last_error


def _call_openai_compatible(client: OpenAI, model: str, system: str, user: str) -> dict:
    print(f"  [llm] calling {model}...", flush=True)
    start = time.time()
    response = _with_retries(lambda: client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
    ))
    elapsed = time.time() - start
    print(f"  [llm] {model} responded in {elapsed:.1f}s", flush=True)
    logfire.info(
        "llm_call",
        model=model,
        elapsed_seconds=elapsed,
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
    )
    return json.loads(response.choices[0].message.content)


def call_planner(system: str, user: str) -> dict:
    return _call_openai_compatible(_nim_planner, config.PLANNER_MODEL, system, user)


def call_executor(system: str, user: str) -> dict:
    return _call_openai_compatible(_groq, config.EXECUTOR_MODEL, system, user)


def call_judge(system: str, user: str) -> dict:
    return _call_openai_compatible(_nim_judge, config.JUDGE_MODEL, system, user)


def call_gemini(model: str, system: str, user: str) -> dict:
    print(f"  [llm] calling {model}...", flush=True)
    start = time.time()
    response = _with_retries(lambda: _gemini.models.generate_content(
        model=model,
        contents=user,
        config={"system_instruction": system, "response_mime_type": "application/json"},
    ))
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
    return json.loads(response.text)
