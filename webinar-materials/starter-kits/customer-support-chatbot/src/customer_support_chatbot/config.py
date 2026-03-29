from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class AppConfig:
    openai_api_key: str
    model_name: str


def load_config() -> AppConfig:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing. Add it in .env or shell environment.")

    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
    return AppConfig(openai_api_key=api_key, model_name=model_name)


def default_paths(project_root: Path) -> dict[str, Path]:
    return {
        "faq_csv": project_root / "data" / "customer_support_faq.csv",
        "chroma_dir": project_root / "data" / "chroma_faq",
    }
