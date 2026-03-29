from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class AppConfig:
    openai_api_key: str
    model_name: str = "gpt-4o-mini"


def load_config() -> AppConfig:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing. Add it in .env or shell environment.")

    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
    return AppConfig(openai_api_key=api_key, model_name=model_name)


def default_project_paths(project_root: Path) -> dict[str, Path]:
    return {
        "resumes_dir": project_root / "data" / "resumes",
        "job_description_path": project_root / "data" / "job_description.txt",
        "output_dir": project_root / "outputs",
    }
