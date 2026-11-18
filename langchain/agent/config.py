from pathlib import Path
import os

from dotenv import load_dotenv


AGENT_DIR = Path(__file__).resolve().parent
DATA_DIR = AGENT_DIR / "runtime_data"
UPLOAD_DIR = DATA_DIR / "uploads"
DATABASE_PATH = DATA_DIR / "research_agent.sqlite3"
STATIC_DIR = AGENT_DIR / "static"

load_dotenv(AGENT_DIR / ".env", override=True)


def get_api_key() -> str:
    return os.getenv("OPENAI_API_KEY", "").strip()


def get_base_url() -> str:
    return os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/")


def get_model() -> str:
    return os.getenv("OPENAI_MODEL", "").strip()


def get_embedding_model() -> str:
    configured = os.getenv("OPENAI_EMBEDDING_MODEL", os.getenv("EMBEDDING_MODEL", "")).strip()
    if configured:
        return configured
    if "siliconflow" in get_base_url().lower():
        return "BAAI/bge-m3"
    return ""


def ensure_runtime_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
