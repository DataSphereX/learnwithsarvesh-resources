# Resume Analyzer Starter Kit

Production-style starter kit version of the webinar notebook.

## What this project does

- Reads all PDF resumes from `data/resumes/`
- Extracts structured candidate info using an LLM
- Extracts required skills from `data/job_description.txt`
- Computes skill match score for each resume
- Saves per-candidate JSON reports and a ranking CSV

## Project structure

- `src/resume_analyzer/`: Reusable package code
- `scripts/run_batch.py`: Batch processing entrypoint
- `data/resumes/`: Put resume PDFs here
- `data/job_description.txt`: Role requirement input
- `outputs/`: Generated reports and ranking

## Quick start

1. Create and activate a virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Configure secrets:
   - Copy `.env.example` to `.env`
   - Add your OpenAI key
4. Add resume PDFs into `data/resumes/`.
5. Update `data/job_description.txt`.
6. Run:
   - `python scripts/run_batch.py`

## Output

- JSON report per resume in `outputs/`
- Ranking CSV at `outputs/candidate_ranking.csv`
