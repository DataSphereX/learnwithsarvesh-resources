from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rag_pdf_summarizer.config import default_paths, load_config
from rag_pdf_summarizer.indexer import build_vectorstore


def main() -> None:
    project_root = PROJECT_ROOT
    paths = default_paths(project_root)

    config = load_config()
    chunk_count = build_vectorstore(
        api_key=config.openai_api_key,
        pdf_dir=paths["pdf_dir"],
        vectorstore_dir=paths["vectorstore_dir"],
    )

    print(f"Built FAISS index with {chunk_count} chunks at {paths['vectorstore_dir']}")


if __name__ == "__main__":
    main()
