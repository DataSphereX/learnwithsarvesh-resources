from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from customer_support_chatbot.chatbot import FAQChatbot
from customer_support_chatbot.config import default_paths, load_config
from customer_support_chatbot.knowledge_base import build_or_load_retriever


def main() -> None:
    project_root = PROJECT_ROOT
    paths = default_paths(project_root)

    config = load_config()
    retriever = build_or_load_retriever(
        api_key=config.openai_api_key,
        persist_dir=paths["chroma_dir"],
        documents=None,
    )

    chatbot = FAQChatbot(api_key=config.openai_api_key, model_name=config.model_name)
    chain = chatbot.build_chain(retriever)

    session_id = "demo_session"
    print("FAQ chatbot ready. Type 'exit' to quit.")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if not question:
            continue

        answer = chatbot.ask(chain, question=question, session_id=session_id)
        print(f"Bot: {answer}\n")


if __name__ == "__main__":
    main()
