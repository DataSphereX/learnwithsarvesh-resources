from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from langchain_community.document_loaders import PyPDFLoader


def load_resume_texts(resumes_dir: Path) -> list[tuple[str, str]]:
    if not resumes_dir.exists():
        raise FileNotFoundError(f"Resume folder not found: {resumes_dir}")

    resume_files = sorted([path for path in resumes_dir.glob("*.pdf") if path.is_file()])
    if not resume_files:
        raise FileNotFoundError(f"No PDF resumes found in: {resumes_dir}")

    resume_texts: list[tuple[str, str]] = []
    for resume_file in resume_files:
        pages = PyPDFLoader(str(resume_file)).load()
        text = "\n".join(page.page_content for page in pages)
        resume_texts.append((resume_file.name, text))

    return resume_texts


def save_result_bundle(output_dir: Path, result: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(result["file_name"]).stem

    result_path = output_dir / f"{stem}.json"
    with result_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)


def save_ranking_csv(output_dir: Path, rows: list[dict]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = output_dir / "candidate_ranking.csv"

    data = [
        {
            "file_name": row["file_name"],
            "name": row["parsed_resume"].get("name", "Unknown"),
            "match_score": row["match_score"],
            "matched_skills": ", ".join(row["matched_skills"]),
            "missing_skills": ", ".join(row["missing_skills"]),
        }
        for row in rows
    ]

    df = pd.DataFrame(data).sort_values("match_score", ascending=False)
    df.to_csv(ranking_path, index=False)
    return ranking_path
