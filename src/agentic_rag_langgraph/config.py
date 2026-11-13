"""One place for every tunable. Environment variables override the local defaults (same names and
defaults as the Mastra repo's src/lib/config.ts)."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "qwen3:8b")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "bge-m3")
EMBEDDING_DIMENSION = 1024
TOP_K = int(os.environ.get("TOP_K", "5"))
PROMPT_VERSION = "v1" if os.environ.get("PROMPT") == "v1" else "v2"
INDEX_PATH = Path(os.environ.get("INDEX_PATH", ROOT / "index" / "eu_ai_act_chunks.json"))
# One model step plus one tool step per LangGraph superstep pair. 20 model steps mirrors the
# AI SDK default stop condition (stepCountIs(20)) found in the @mastra/core bundle.
RECURSION_LIMIT = 41
