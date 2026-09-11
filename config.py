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
NIM_API_KEY = _required("NIM_API_KEY")

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_EXECUTOR_API_KEY = _required("GROQ_EXECUTOR_API_KEY")
# Eval-only, so kept optional: the API/UI entrypoints never call the judge,
# and shouldn't need this key just to start up. If you're on a paid Groq
# plan, point this at the same key as GROQ_EXECUTOR_API_KEY, see README.
GROQ_JUDGE_API_KEY = os.getenv("GROQ_JUDGE_API_KEY", "")

GEMINI_API_KEY = _required("GEMINI_API_KEY")
TAVILY_API_KEY = _required("TAVILY_API_KEY")
NEON_DSN = os.getenv("NEON_DSN", "")
LOGFIRE_TOKEN = os.getenv("LOGFIRE_TOKEN", "")

PLANNER_MODEL = "nvidia/nemotron-3-super-120b-a12b"
EXECUTOR_MODEL = "openai/gpt-oss-120b"
CRITIC_MODEL = "gemini-3.5-flash-lite"
SYNTHESIZER_MODEL = "gemini-3.5-flash"
JUDGE_MODEL = "openai/gpt-oss-120b"

REQUIRED_FIELDS = ["what_it_does", "funding_ownership", "recent_news", "competitors", "risks"]

MAX_REPLAN_CYCLES = 3
MAX_TOOL_CALLS = 15
MAX_WALL_CLOCK_SECONDS = 8 * 60
MEMORY_CACHE_DAYS = 7
FUZZY_MATCH_THRESHOLD = 90
