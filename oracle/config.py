import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY    = os.environ["OPENAI_API_KEY"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
GEMINI_API_KEY    = os.environ["GEMINI_API_KEY"]
DEEPSEEK_API_KEY  = os.getenv("DEEPSEEK_API_KEY", "")

OPENAI_MODEL    = os.getenv("OPENAI_MODEL",    "gpt-4o")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
GEMINI_MODEL    = os.getenv("GEMINI_MODEL",    "gemini-2.0-flash")
DEEPSEEK_MODEL  = os.getenv("DEEPSEEK_MODEL",  "deepseek-v4-pro")

AMBIGUITY_THRESHOLD = float(os.getenv("AMBIGUITY_THRESHOLD", "0.5"))
