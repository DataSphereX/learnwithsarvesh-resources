# TechXConf 2026 - Agentic Data Lineage Demo

Thank you for attending my session at **TechXConf 2026**! This folder contains
the slides and the full runnable code for the **Agentic Data Lineage
Pipeline** demo, so you can try it yourself and build on it.

- **Slides:** [Sarveshwaran-Meet-Chintu-TechXConf-2026.pdf](Sarveshwaran-Meet-Chintu-TechXConf-2026.pdf)
- **Code:** [Code/](Code/)
- **Speaker:** Sarveshwaran, *Learn with Sarvesh*

## What the demo does

The demo shows how an agentic workflow can inspect SQL pipelines, identify
source-to-target lineage, classify transformations, detect potential PII,
route work to an appropriate model tier, validate the result, and write a
reviewable metadata record to a catalog.

Key ideas you saw in the talk:

- **Scrubbing first** - SQL literals and credential-like values are removed
  before anything is sent to a model.
- **Cost-aware routing** - small, mid, and large model tiers are chosen based
  on pipeline complexity.
- **Validation** - a second agent checks the extracted lineage and assigns a
  confidence score.
- **Human in the loop** - the workflow pauses for a human decision only when
  confidence is below the commit threshold.
- **Auditable output** - every result, with its estimated token cost, is
  recorded in `catalog.json`.
- **Works offline** - cached model responses let you run the whole demo
  without an API key or network access.

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
- `cached_responses.json` — responses used by offline mode, so the demo runs
  without network access or an API key.
- `archive/` — earlier cached responses and catalog snapshots retained for
  reference.
- `agentic_lineage_README.md.docx` — a longer document version of the demo
  notes.
- `commands to run.txt` — a compact command reference.

## How the workflow works

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

The output is metadata, not a replacement for a SQL execution engine. The SQL
files are examples for lineage extraction and are not executed by the Python
demo.

## Requirements

- Python 3.10 or newer
- The packages listed in `Code/requirements.txt`:
  - `langgraph`
  - `openai`
  - `python-dotenv`
- An OpenAI API key, **only** for live mode

## Setup

First, clone the repository:

```bash
git clone https://github.com/DataSphereX/learnwithsarvesh-resources.git
cd learnwithsarvesh-resources/TechXConf2026/Code
```

### Windows (PowerShell)

```powershell
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

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the demo

### Offline mode (start here)

Offline mode uses `cached_responses.json` and makes no API calls, so you don't
need an API key:

```bash
python agentic_lineage_demo.py --offline
```

To be asked for approval at the human gate (this is the run from the talk):

```bash
python agentic_lineage_demo.py --offline --interactive
```

To see what happens when the human gate rejects a result:

```bash
python agentic_lineage_demo.py --offline --reject
```

### Live mode

Live mode requires `OPENAI_API_KEY` in the environment or in a `.env` file
next to `agentic_lineage_demo.py`:

```text
OPENAI_API_KEY=replace_with_your_key
```

Run the readiness check first. It validates the API key and makes a small
request for each configured model tier:

```bash
python agentic_lineage_demo.py --check
```

Then run:

```bash
python agentic_lineage_demo.py
```

Live requests may incur provider charges. The script uses these model
settings, each of which can be overridden with an environment variable:

| Tier | Default model | Environment variable |
|---|---|---|
| Small | `gpt-4o-mini` | `OPENAI_MODEL_SMALL` |
| Mid | `gpt-4o` | `OPENAI_MODEL_MID` |
| Large | `gpt-4.1` | `OPENAI_MODEL_LARGE` |

## Command-line options

| Option | Purpose |
|---|---|
| `--offline` | Replay responses from `cached_responses.json` without network access. |
| `--interactive` | Ask for approval at the human gate instead of automatically accepting. |
| `--reject` | Simulate a rejection at the human gate. |
| `--check` | Validate the environment and test each configured live model, then exit. |

The options can be combined where appropriate, for example
`--offline --interactive`.

## Understanding the catalog

Each catalog record describes one processed pipeline. Important fields include:

- `source_tables` — upstream tables discovered in the SQL.
- `transformations` — output columns and their logic.
- `type` — transformation category:

  | Type | Meaning |
  |---|---|
  | `passthrough` | Copied as-is, no computation |
  | `aggregate` | Computed via `SUM`/`COUNT`/`AVG`/etc. over rows |
  | `derived` | Computed via some expression or formula |
  | `label` | A computed boolean or categorical flag |

- `pii_risk_columns` — columns that may contain personally identifiable
  information or sensitive attributes.
- `confidence` — validation confidence for the extracted lineage.
- `tier` — model tier selected by the complexity router.
- `cost_usd` — estimated model cost for the run.
- `catalog_status` — whether the record was automatically committed or
  required human review.

> **Tip:** The SQL examples use CTEs (Common Table Expressions) — named
> sub-queries defined with `WITH ... AS ( ... )` and used like a temporary
> table inside one query.

To inspect the catalog, open `catalog.json` in any editor, or on Windows:

```powershell
Get-Content .\catalog.json | ConvertFrom-Json | ConvertTo-Json -Depth 10
```

On macOS / Linux:

```bash
python -m json.tool catalog.json
```

## Try it with your own SQL

The script processes the three files listed in the `files` list inside
`main()` in `agentic_lineage_demo.py`. To try your own pipeline:

1. Add your `.sql` file to `Code/pipelines/` (use synthetic or approved
   sample SQL only).
2. Add or swap its path in the `files` list in `agentic_lineage_demo.py`.
3. Run in live mode, since the offline cache only covers the three examples.
4. Inspect the new record in `catalog.json`.

Live runs save model responses to `cached_responses.json`, so you can replay
your pipeline with `--offline` afterwards.

Do not place API keys in `cached_responses.json`, `catalog.json`, source code,
or committed files. Keep `.env` local and untracked.

## Troubleshooting

### `ModuleNotFoundError`

Activate the virtual environment and install the requirements:

```bash
python -m pip install -r requirements.txt
```

### `OPENAI_API_KEY not set`

Use offline mode, which needs no key:

```bash
python agentic_lineage_demo.py --offline
```

For live mode, set the key in your current terminal session:

```powershell
# Windows PowerShell
$env:OPENAI_API_KEY = "replace_with_your_key"
```

```bash
# macOS / Linux
export OPENAI_API_KEY="replace_with_your_key"
```

### Offline cache entry is missing

The cache must contain an entry for each pipeline. Run a live execution once
to rebuild it, or restore the original `cached_responses.json` from the
repository.

### Garbled symbols in the Windows terminal

Set UTF-8 output encoding before running:

```powershell
$env:PYTHONIOENCODING = "utf-8"
```

### The human gate does not appear

Use `--interactive`, and confirm that the selected scenario produces a
low-confidence validation result:

```bash
python agentic_lineage_demo.py --offline --interactive
```

## Safety and responsible use

- Never commit real API keys, `.env` files, customer data, or production
  credentials.
- Use synthetic or approved sample SQL.
- Treat model-generated lineage as an assisted result that requires validation.
- Review PII flags and low-confidence records before publishing metadata.
- Confirm model names and pricing before using live mode in a production
  workflow; the prices in the script are illustrative defaults.

## Taking it further

Ideas to explore after the session:

- Where would human review, data governance, and production controls sit in
  your own data platform?
- How would you tune the complexity router and confidence threshold for your
  pipelines?
- Could the catalog output feed an existing metadata tool in your organisation?

Questions or feedback? Reach out via *Learn with Sarvesh* — I'd love to hear
what you build with this.
