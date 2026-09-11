import os
from dotenv import load_dotenv

load_dotenv()


def _required(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise RuntimeError(
            f"missing required environment variable: {key} (check your .env file)"
        )
    return value


NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NIM_PLANNER_API_KEY = _required("NIM_PLANNER_API_KEY")
# Only needed for eval/judge.py, kept optional so the API/UI don't require an
# eval-only account just to import config.
NIM_JUDGE_API_KEY = os.getenv("NIM_JUDGE_API_KEY", "")

GROQ_API_KEY = _required("GROQ_API_KEY")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

GEMINI_API_KEY = _required("GEMINI_API_KEY")
TAVILY_API_KEY = _required("TAVILY_API_KEY")
NEON_DSN = os.getenv("NEON_DSN", "")
LOGFIRE_TOKEN = os.getenv("LOGFIRE_TOKEN", "")

PLANNER_MODEL = "meta/llama-3.1-8b-instruct"
EXECUTOR_MODEL = "openai/gpt-oss-120b"
CRITIC_MODEL = "gemini-3.5-flash"
SYNTHESIZER_MODEL = "gemini-3.5-flash"
JUDGE_MODEL = "deepseek-ai/deepseek-v4-flash-0731"

REQUIRED_FIELDS = ["what_it_does", "funding_ownership", "recent_news", "competitors", "risks"]

MAX_REPLAN_CYCLES = 3
MAX_TOOL_CALLS = 15
MAX_WALL_CLOCK_SECONDS = 8 * 60
MEMORY_CACHE_DAYS = 7
FUZZY_MATCH_THRESHOLD = 90
