# CDC_Ford_May16_2026 — Student Guide

This folder contains materials used in the CDC / Ford session (May 16, 2026). This README is written for beginners and explains what each file is, what it does, and simple steps to run things on your computer.

**What is here**
- `LangGraph Agent Router.ipynb` — A Jupyter Notebook. It contains code cells you can run step-by-step to explore agent routing concepts. Open it with Jupyter or VS Code and run the cells.
- `questions to ask.txt` — Plain text file of suggested questions to use during a demo or discussion.
- `sales.csv` — A sample dataset of sales records used by the notebook and the Streamlit app.
- `streamlit_sales_agent_app.py` — A small Streamlit app that reads `sales.csv` and shows a simple interactive demo.

Prerequisites (very simple)
- Python 3.8 or newer installed. You can check with:

```bash
python --version
```

- (Recommended) Create and use a virtual environment so the demo packages don't affect other Python projects.

Quick setup — Windows PowerShell (example)

```powershell
# create a virtual environment (run once)
python -m venv .venv

# activate the virtual environment
.\.venv\Scripts\Activate.ps1

# upgrade pip and install minimal packages
pip install --upgrade pip
pip install streamlit pandas jupyter
```

Quick setup — macOS / Linux (example)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install streamlit pandas jupyter
```

How to run the Streamlit app
1. Activate your virtual environment (see steps above).
2. From this folder run:

```bash
streamlit run streamlit_sales_agent_app.py
```

3. Streamlit will open a browser tab (or show a local URL) where you can interact with the app.

How to open and run the Jupyter Notebook
1. Activate your virtual environment.
2. Start Jupyter from this folder:

```bash
jupyter notebook "LangGraph Agent Router.ipynb"
```

3. The notebook will open in your browser. Run cells in order (use the ▶ Run button). If any cell needs extra packages, the output will usually tell you which package is missing.

Quick way to view the CSV file

```bash
python -c "import pandas as pd; print(pd.read_csv('sales.csv').head())"
```

Notes and troubleshooting (for beginners)
- If a command says a package is missing, install it with `pip install <package-name>` while your virtual environment is active.
- If Streamlit doesn't open a browser automatically, copy the local URL printed in the terminal (http://localhost:8501) into your browser.
- If you use VS Code, you can open the notebook directly in the editor and run cells there.

What to try first (suggested)
- Open `sales.csv` to understand the columns (date, product, sales, region, etc.).
- Run the Streamlit app and try interacting with the controls.
- Open the notebook and run the first few cells to see how the data is loaded and used.

If you need help
- Ask your instructor or lab assistant and show the error message you see.
- You can also add notes in `questions to ask.txt` as you try things.

Have fun exploring — this folder is meant for learning and experimenting.
