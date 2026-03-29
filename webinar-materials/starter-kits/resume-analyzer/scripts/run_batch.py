from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from resume_analyzer.config import default_project_paths, load_config
from resume_analyzer.io import load_resume_texts, save_ranking_csv, save_result_bundle
from resume_analyzer.pipeline import ResumeAnalyzer


def main() -> None:
    project_root = PROJECT_ROOT
    paths = default_project_paths(project_root)

    config = load_config()
    analyzer = ResumeAnalyzer(config)

    job_description = paths["job_description_path"].read_text(encoding="utf-8")
    required_skills = analyzer.extract_required_skills(job_description)

    print(f"Required skills extracted: {required_skills}")

    results = []
    for file_name, resume_text in load_resume_texts(paths["resumes_dir"]):
        print(f"Processing {file_name}...")
        result = analyzer.process_resume(file_name, resume_text, required_skills)
        payload = result.model_dump()
        save_result_bundle(paths["output_dir"], payload)
        results.append(payload)

    ranking_path = save_ranking_csv(paths["output_dir"], results)
    print(f"Saved ranking: {ranking_path}")


if __name__ == "__main__":
    main()
