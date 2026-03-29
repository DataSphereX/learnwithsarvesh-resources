from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from customer_support_chatbot.config import default_paths, load_config
from customer_support_chatbot.knowledge_base import build_or_load_retriever, load_faq_documents


def main() -> None:
    project_root = PROJECT_ROOT
    paths = default_paths(project_root)

    config = load_config()
    documents = load_faq_documents(paths["faq_csv"])
    build_or_load_retriever(
        api_key=config.openai_api_key,
        persist_dir=paths["chroma_dir"],
        documents=documents,
    )

    print(f"Indexed {len(documents)} FAQ entries into {paths['chroma_dir']}")


if __name__ == "__main__":
    main()
