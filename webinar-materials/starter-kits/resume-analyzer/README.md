# Resume Analyzer Starter Kit

Production-style starter kit version of the webinar notebook.

This starter kit now supports both:

- Batch processing via `scripts/run_batch.py`
- Interactive UI via `AI Resume Analyzer streamlit app.py`

## What this project does

- Reads all PDF resumes from `data/resumes/`
- Extracts structured candidate info using an LLM
- Extracts required skills from `data/job_description.txt`
- Computes skill match score for each resume
- Saves per-candidate JSON reports and a ranking CSV

## Project structure

- `src/resume_analyzer/`: Reusable package code
- `scripts/run_batch.py`: Batch processing entrypoint
- `AI Resume Analyzer streamlit app.py`: Streamlit UI for live multi-resume analysis
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
4. Choose one run mode:
   - Batch mode:
     - Add resume PDFs into `data/resumes/`
     - Update `data/job_description.txt`
     - Run `python scripts/run_batch.py`
   - Streamlit mode:
     - Run `streamlit run "AI Resume Analyzer streamlit app.py"`
     - Paste job description in the UI
     - Upload one or more resume PDFs
     - Click **Analyze Resumes**

## Output

- Batch mode:
   - JSON report per resume in `outputs/`
   - Ranking CSV at `outputs/candidate_ranking.csv`
- Streamlit mode:
   - On-screen leaderboard and ranking table
   - Downloadable ranking CSV
   - Downloadable per-candidate JSON reports
