from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf_documents(pdf_dir: Path):
    pdf_files = sorted([path for path in pdf_dir.glob("*.pdf") if path.is_file()])
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {pdf_dir}")

    documents = []
    for pdf_file in pdf_files:
        pages = PyPDFLoader(str(pdf_file)).load()
        documents.extend(pages)
    return documents


def build_vectorstore(api_key: str, pdf_dir: Path, vectorstore_dir: Path) -> int:
    documents = load_pdf_documents(pdf_dir)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(vectorstore_dir))
    return len(chunks)
