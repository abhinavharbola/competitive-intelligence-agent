import os
from dotenv import load_dotenv

load_dotenv()

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NIM_PLANNER_API_KEY = os.environ["NIM_PLANNER_API_KEY"]
NIM_JUDGE_API_KEY = os.environ["NIM_JUDGE_API_KEY"]

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]
NEON_DSN = os.environ.get("NEON_DSN", "")
LOGFIRE_TOKEN = os.environ.get("LOGFIRE_TOKEN", "")

PLANNER_MODEL = "meta/llama-3.1-8b-instruct"
EXECUTOR_MODEL = "openai/gpt-oss-120b"
CRITIC_MODEL = "gemini-3.5-flash"
SYNTHESIZER_MODEL = "gemini-3.5-flash"
JUDGE_MODEL = "qwen/qwen2.5-7b-instruct"

REQUIRED_FIELDS = ["what_it_does", "funding_ownership", "recent_news", "competitors", "risks"]

MAX_REPLAN_CYCLES = 3
MAX_TOOL_CALLS = 15
MAX_WALL_CLOCK_SECONDS = 8 * 60
MEMORY_CACHE_DAYS = 7
FUZZY_MATCH_THRESHOLD = 90
