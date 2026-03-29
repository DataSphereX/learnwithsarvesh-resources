from __future__ import annotations

from pathlib import Path

import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


def load_faq_documents(faq_csv: Path) -> list[Document]:
    if not faq_csv.exists():
        raise FileNotFoundError(f"Missing FAQ CSV: {faq_csv}")

    df = pd.read_csv(faq_csv)
    docs = [
        Document(
            page_content=f"Question: {row['question']}\nAnswer: {row['answer']}",
            metadata={
                "source": "faq_csv",
                "row": int(index),
                "region": str(row.get("region", "global")),
                "channel": str(row.get("channel", "web")),
                "issue": str(row.get("issue", "general")),
            },
        )
        for index, row in df.iterrows()
    ]
    return docs


def build_or_load_retriever(
    api_key: str,
    persist_dir: Path,
    documents: list[Document] | None = None,
):
    embedder = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    persist_dir.mkdir(parents=True, exist_ok=True)

    if documents:
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embedder,
            persist_directory=str(persist_dir),
        )
    else:
        vectorstore = Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embedder,
        )

    return vectorstore.as_retriever(search_kwargs={"k": 4})
