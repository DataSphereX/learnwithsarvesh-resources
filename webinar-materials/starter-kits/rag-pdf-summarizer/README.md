# PDF Document Summarizer (RAG) Starter Kit

Structured project starter based on the webinar notebook.

## What this project does

- Loads PDF files from `data/pdfs/`
- Splits content into chunks
- Builds and saves FAISS vector index
- Runs an interactive RAG Q&A CLI over indexed PDFs

## Project structure

- `src/rag_pdf_summarizer/`: Reusable RAG modules
- `scripts/build_index.py`: Build FAISS index from PDFs
- `scripts/chat_cli.py`: Ask questions over indexed PDFs
- `data/pdfs/`: Input PDFs
- `data/vectorstore/`: Saved FAISS artifacts

## Quick start

1. Create and activate a virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Configure secrets:
   - Copy `.env.example` to `.env`
   - Add your OpenAI key
4. Add one or more PDF files to `data/pdfs/`.
5. Build index:
   - `python scripts/build_index.py`
6. Chat with your documents:
   - `python scripts/chat_cli.py`
