import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True))

BASE_URL = os.getenv("LITELLM_BASE_URL", "http://a6k2.dgx:34000/v1")
API_KEY = os.getenv("LITELLM_API_KEY", "") or "dummy_key"
MODEL_NAME = os.getenv("MODEL_NAME", "qwen3-32b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))

GDELT_BASE = os.getenv("GDELT_BASE", "https://api.gdeltproject.org/api/v2/doc/doc")
STOOQ_BASE = os.getenv("STOOQ_BASE", "https://stooq.com/q/d/l/")

HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "20"))
HTTP_RETRIES = int(os.getenv("HTTP_RETRIES", "3"))

EVENT_WINDOW_DAYS = int(os.getenv("EVENT_WINDOW_DAYS", "1"))
MAX_ARTICLES = int(os.getenv("MAX_ARTICLES", "30"))