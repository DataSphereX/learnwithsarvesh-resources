# Student Notes — LangGraph Agent Router (Beginner Friendly)

Purpose
- These notes explain the ideas and code used in `LangGraph Agent Router.ipynb` in simple terms. They help you understand how the multi-agent system works and what each part of the notebook does.

Overview (high level)
- Problem: An analyst sees revenue drop and needs fast insight.
- Solution: An intelligent router (the "brain") reads a plain-English question, picks a specialist agent, the specialist analyzes the data, and a Business Translator turns technical results into executive-friendly recommendations.
- Result: Analysts get answers in seconds instead of hours.

Key Concepts (easy explanations)

- Router (Intent Detection)
  - Role: Read the analyst's question and decide which specialist should answer.
  - How it works: The notebook calls an LLM (GPT-4o-mini) to classify intent into routes like `anomaly_agent`, `trend_agent`, `breakdown_agent`, `recommendation_agent`, or `summary_agent`.
  - Fallback: If the LLM is unavailable, a simple keyword-based function checks words in the question and chooses a route.

- Specialist Agents (Focused Workers)
  - Each agent has a single responsibility and returns technical answers.
  - `anomaly_agent`: Finds when revenue dropped, how severe it is, and which month was worst.
  - `trend_agent`: Compares revenue over time and lists products that gained or lost traction.
  - `breakdown_agent`: Splits revenue by region and top sales reps to show where the problem lives.
  - `recommendation_agent`: Suggests actions based on what is working and what is not (e.g., which product or region to prioritize).
  - `summary_agent`: Gives a short technical overview (totals, averages, top product/region).

- Business Translator (Humanizer)
  - Role: Convert technical outputs from specialists into clear, executive-ready language.
  - Output: An executive summary, key findings, recommended actions, and follow-up questions.
  - Why it's important: Executives want short, actionable insights, not raw technical tables.

How the Notebook Demonstration Works (step-by-step)

1. Load data
   - The notebook loads `sales.csv` into a DataFrame and creates helper columns like `month` and `quarter`.
2. Quick dashboard view
   - It prints monthly and quarterly revenue summaries so you can see the drop in Q2.
3. Routing logic
   - The notebook defines `detect_route_with_llm()` which asks the LLM to return a `ROUTE:` and `REASONING:` string.
   - If the LLM fails, `detect_route_fallback()` uses keywords to pick a route.
4. Specialist code
   - Each specialist function receives a `state` (question + placeholders) and fills `state['answer']` and `state['route']` with technical findings.
5. Translator
   - `business_translator_agent()` sends the technical answer to the LLM and gets back a human-friendly formatted result.
6. Demo flow
   - `run_agent_demo(question)` runs the router, then the chosen specialist, then the translator, and returns a final state with the executive answer.

Important Code Details (plain language)

- State object
  - A small dictionary with keys: `question`, `route`, `answer`, `reasoning`.
  - The notebook defines `AnalystState` as a typing hint (helps with clarity but is optional).

- LLM prompts
  - The notebook uses clear instructions for the LLM so it responds in a predictable format (e.g., lines beginning with `ROUTE:` and `REASONING:`). This makes parsing easy.

- Error handling and fallback
  - If API calls fail or packages are missing, the code prints warnings and uses the keyword fallback so the demo can still run.

Security & Practical Notes (for beginners)
- API keys: The notebook looks for `OPENAI_API_KEY` in a `.env` file. Keep keys private and do not share them.
- Costs: Calling an LLM (OpenAI) may cost money. Use small models for demos or run the notebook in offline/demo mode by using the fallback routing.

How to run the notebook (simple)
1. Install Python and the basic packages (example):

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
pip install pandas python-dotenv openai
```

2. (Optional) If you have an OpenAI API key, create a `.env` file in the same folder with:

```text
OPENAI_API_KEY=sk-...your key...
```

3. Open `LangGraph Agent Router.ipynb` in Jupyter or VS Code and run cells in order. The demo shows four typical analyst questions and how the pipeline answers them.

Suggested Beginner Exercises
- Exercise 1 — Run with fallback only:
  - Temporarily rename or remove your `.env` key so LLM calls fail and the notebook uses keyword fallback.
  - Observe which routes are chosen and why.
- Exercise 2 — Modify the fallback keywords:
  - Add or remove words in `detect_route_fallback()` and test how routing changes for different questions.
- Exercise 3 — Inspect specialist outputs:
  - Print `monthly_summary`, `product_trend`, or `region_perf` inside each specialist to see raw numbers.
- Exercise 4 — Change the demo questions:
  - Edit `demo_workflow` questions to ones from your own company or a dataset you like and run the demo.
- Exercise 5 — Try the Business Translator alone:
  - Set `state['answer']` to a mock technical output string and call `business_translator_agent(state)` to see the translation.

Common Troubleshooting
- Missing packages: Run `pip install pandas python-dotenv openai`.
- No OpenAI key: If you don't want to use the API, rely on the fallback routing to see the rest of the demo.
- Streamlit app conflict: The folder also contains `streamlit_sales_agent_app.py`. That file is separate — you can run it with `streamlit run streamlit_sales_agent_app.py` if you want a web demo.

Questions to think about (discussion prompts)
- What is the advantage of separating the router from the specialists?
- How would you add a new specialist (for example, a forecasting agent)? What parts of the code would you change?
- When could the LLM routing fail, and how can we make the fallback more robust?
- How does the Business Translator help with decision-making beyond raw numbers?

Next steps (if you want more help)
- I can create guided exercise notebooks that implement the exercises above.
- I can also add inline comments to the notebook to explain each code block line-by-line.

---
These notes are meant to make the notebook approachable. If you'd like, I will now add inline comments to the notebook's code cells or create a step-by-step exercise notebook — tell me which you prefer.
