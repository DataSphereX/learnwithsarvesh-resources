from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rag_pdf_summarizer.config import default_paths, load_config
from rag_pdf_summarizer.qa import PDFRAG


def main() -> None:
    project_root = PROJECT_ROOT
    paths = default_paths(project_root)

    config = load_config()
    rag = PDFRAG(api_key=config.openai_api_key, model_name=config.model_name)
    chain = rag.build_chain(paths["vectorstore_dir"])

    print("PDF RAG assistant ready. Type 'exit' to quit.")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if not question:
            continue

        answer = chain.invoke({"input": question})
        print(f"Bot: {answer}\n")


if __name__ == "__main__":
    main()
