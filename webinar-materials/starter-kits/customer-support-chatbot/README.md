# Customer Support Chatbot Starter Kit

Projectized version of the webinar RAG chatbot notebook.

## What this project does

- Builds an FAQ dataset (`data/customer_support_faq.csv`)
- Embeds FAQs and stores vectors in persistent Chroma DB
- Runs a conversational support chatbot with session memory

## Project structure

- `src/customer_support_chatbot/`: Reusable chatbot code
- `data/generate_faq.py`: Generates sample 500-row FAQ data
- `scripts/build_index.py`: Creates Chroma vector index
- `scripts/chat_cli.py`: Interactive chatbot CLI

## Quick start

1. Create and activate a virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Configure secrets:
   - Copy `.env.example` to `.env`
   - Add your OpenAI key
4. Generate FAQ data:
   - `python data/generate_faq.py`
5. Build vector index:
   - `python scripts/build_index.py`
6. Run chatbot:
   - `python scripts/chat_cli.py`

## Notes

- Chroma index is stored in `data/chroma_faq/`.
- Replace generated FAQ CSV with your domain-specific support data.
