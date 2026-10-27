# TechXConf 2026 - Agentic Data Lineage Demo

This folder contains the presentation material and runnable code for the
TechXConf 2026 demonstration **Agentic Data Lineage Pipeline**.

The demo shows how an agentic workflow can inspect SQL pipelines, identify
source-to-target lineage, classify transformations, detect potential PII,
route work to an appropriate model tier, validate the result, and write a
reviewable metadata record to a catalog.

The workflow is intentionally designed for a live technical presentation:

- It can run completely offline using cached model responses.
- It scrubs SQL before sending content to a model.
- It uses small, medium, and large model tiers based on pipeline complexity.
- It pauses for a human decision when confidence is below the commit threshold.
- It records the result and estimated token cost in `catalog.json`.

## Contents

```text
TechXConf2026/
|-- README.md
|-- Sarveshwaran-Meet-Chintu-TechXConf-2026.pdf
`-- Code/
    |-- agentic_lineage_demo.py
    |-- agentic_lineage_README.md.docx
    |-- commands to run.txt
    |-- requirements.txt
    |-- catalog.json
    |-- cached_responses.json
    |-- archive/
    `-- pipelines/
        |-- simple_customer_view.sql
        |-- daily_revenue_rollup.sql
        `-- customer_churn_features.sql
```

### Presentation

`Sarveshwaran-Meet-Chintu-TechXConf-2026.pdf` contains the talk slides.

### Demo application

`Code/agentic_lineage_demo.py` is the executable demonstration. It processes
the SQL files in `Code/pipelines/`, calls the model when live mode is enabled,
and updates the metadata catalog.

### SQL examples

The three SQL files deliberately represent increasing levels of complexity:

1. **`simple_customer_view.sql`** — a straightforward view with mostly
   passthrough columns and a small amount of PII risk.
2. **`daily_revenue_rollup.sql`** — joins billing and account data, converts
   currency, and performs daily aggregations.
3. **`customer_churn_features.sql`** — combines account, subscription,
   support, sentiment, and product-usage data to build model features.

### Supporting files

- `catalog.json` — the generated lineage catalog containing source tables,
  transformations, PII-risk columns, confidence, model tier, and estimated
  cost.
- `cached_responses.json` — responses used by offline mode, so the demo does
  not need network access or an API key on stage.
- `archive/` — earlier cached responses and catalog snapshots retained for
  reference.
- `agentic_lineage_README.md.docx` — the original longer document version of
  the demo notes.
- `commands to run.txt` — a compact rehearsal reference for Windows PowerShell.

## How the workflow works

The code implements the following flow:

```text
SQL files
   │
   ▼
Parser and scrubber
   │  Removes literals and credential-like values before model calls
   ▼
Complexity router
   │  Selects small, mid, or large model tier
   ▼
Extraction agent
   │  Identifies sources, targets, transformations, and PII risk
   ▼
Validation agent
   │  Checks the extracted lineage and assigns confidence
   ▼
Conditional human gate
   │  Requests approval only when confidence is below the threshold
   ▼
Metadata catalog
```

The output is intended to be metadata, not a replacement for a SQL
execution engine. The SQL files are examples for lineage extraction and are
not executed by the Python demo.

## Requirements

- Python 3.10 or newer
- The packages listed in `Code/requirements.txt`:
  - `langgraph`
  - `openai`
  - `python-dotenv`
- An OpenAI API key only for live mode

## Setup on Windows

Open PowerShell and run:

```powershell
cd "D:\Mentoring\learwithsarvesh\git_repos_public\learnwithsarvesh-resources\TechXConf2026\Code"

py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
$env:PYTHONIOENCODING = "utf-8"
```

If PowerShell prevents activation, either enable script execution for the
current user or invoke the environment's Python executable directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the demo

### Offline rehearsal (recommended)

Offline mode uses `cached_responses.json` and does not make API calls:

```powershell
python agentic_lineage_demo.py --offline
```

For a rehearsal that exercises the approval prompt:

```powershell
python agentic_lineage_demo.py --offline --interactive
```

To rehearse rejecting the human-gate decision:

```powershell
python agentic_lineage_demo.py --offline --reject
```

Run the offline scenarios before the presentation. This verifies that the
environment, cache, terminal encoding, and human-gate behavior are ready
without depending on network connectivity.

### Live mode

Live mode requires `OPENAI_API_KEY` in the environment or in a `.env` file
next to `agentic_lineage_demo.py`:

```text
OPENAI_API_KEY=replace_with_your_key
```

Then run:

```powershell
python agentic_lineage_demo.py
```

The script uses these model settings, each of which can be overridden with an
environment variable:

| Tier | Default model | Environment variable |
|---|---|---|
| Small | `gpt-4o-mini` | `OPENAI_MODEL_SMALL` |
| Mid | `gpt-4o` | `OPENAI_MODEL_MID` |
| Large | `gpt-4.1` | `OPENAI_MODEL_LARGE` |

Use the readiness check before attempting a live run:

```powershell
python agentic_lineage_demo.py --check
```

The check validates the API key and makes a small request for each configured
model tier. Live requests may incur provider charges.

## Command-line options

| Option | Purpose |
|---|---|
| `--offline` | Replay responses from `cached_responses.json` without network access. |
| `--interactive` | Ask for approval at the human gate instead of automatically accepting. |
| `--reject` | Rehearse a rejection at the human gate. |
| `--check` | Validate the environment and test each configured live model, then exit. |

The options can be combined where appropriate. For example,
`--offline --interactive` is useful for a no-network rehearsal of the
approval flow.

## Understanding the catalog

Each catalog record describes one processed pipeline. Important fields include:

- `source_tables` — upstream tables discovered in the SQL.
- `transformations` — output columns and their logic.
- `type` — transformation category such as `passthrough`, `aggregate`,
  `derived`, or `label`.
- `pii_risk_columns` — columns that may contain personally identifiable
  information or sensitive attributes.
- `confidence` — validation confidence for the extracted lineage.
- `tier` — model tier selected by the complexity router.
- `cost_usd` — estimated model cost for the run.
- `catalog_status` — whether the record was automatically committed or
  required human review.

To inspect the catalog from PowerShell:

```powershell
Get-Content .\catalog.json
```

To format it as structured JSON:

```powershell
Get-Content .\catalog.json | ConvertFrom-Json | ConvertTo-Json -Depth 10
```

## Rebuilding the offline cache

Offline mode is only as complete as the cache. If the SQL examples or prompts
change, run one successful live execution to refresh the cached responses,
then verify the offline path:

```powershell
python agentic_lineage_demo.py
python agentic_lineage_demo.py --offline
```

Do not place API keys in `cached_responses.json`, `catalog.json`, source code,
or committed documentation. Keep `.env` local and untracked.

## Troubleshooting

### `ModuleNotFoundError`

Activate the virtual environment and install the requirements:

```powershell
python -m pip install -r requirements.txt
```

### `OPENAI_API_KEY not set`

Use offline mode for a presentation:

```powershell
python agentic_lineage_demo.py --offline
```

For live mode, set the key in the current PowerShell session:

```powershell
$env:OPENAI_API_KEY = "replace_with_your_key"
```

### Offline cache entry is missing

The cache must contain an entry for each demo pipeline. Run a live execution
once to rebuild it, or restore the known-good `cached_responses.json` supplied
with this folder.

### Garbled symbols in the terminal

Set UTF-8 output encoding before running:

```powershell
$env:PYTHONIOENCODING = "utf-8"
```

### The human gate does not appear

Use `--interactive`, and confirm that the selected scenario produces a
low-confidence validation result:

```powershell
python agentic_lineage_demo.py --offline --interactive
```

## Safety and responsible use

- Never commit real API keys, `.env` files, customer data, or production
  credentials.
- Use synthetic or approved sample SQL for demonstrations.
- Treat model-generated lineage as an assisted result that requires validation.
- Review PII flags and low-confidence records before publishing metadata.
- Confirm model names and pricing before using live mode in a production
  workflow; the prices in the script are illustrative defaults.
- Offline mode is the safest choice for a stage demonstration because it
  avoids network calls, unexpected provider responses, and API charges.

## Suggested presentation flow

1. Introduce the problem: lineage extraction is repetitive and difficult to
   keep current.
2. Show the three SQL examples and their increasing complexity.
3. Run `python agentic_lineage_demo.py --offline`.
4. Point out the scrubber, model-tier routing, validation, and cost output.
5. Run `python agentic_lineage_demo.py --offline --interactive`.
6. Approve or reject the low-confidence result and inspect `catalog.json`.
7. Discuss where human review, data governance, and production controls would
   be required.
