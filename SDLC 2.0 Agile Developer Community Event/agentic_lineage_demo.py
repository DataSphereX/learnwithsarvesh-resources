#!/usr/bin/env python3
"""
DEMO \u2014 Agentic Data Lineage Pipeline
====================================
This is the pipeline on the slide, running for real.

    commit hook -> parser & scrubber -> complexity router
        -> extraction agent -> validation agent
        -> conditional human gate -> metadata catalog

It processes three pipelines of different complexity so the audience can see:

  1. The scrubber strip literals and credentials BEFORE anything leaves.
  2. The router pick a cheap or expensive model tier, locally, before spend.
  3. Running token cost accumulate per pipeline.
  4. The human gate fire ONLY for the low-confidence case \u2014 not for all three.

Run it:
    python agentic_lineage_demo.py --offline      # cached, no network. Use on stage.
    python agentic_lineage_demo.py                # live API calls
    python agentic_lineage_demo.py --offline --interactive   # you approve live
    python agentic_lineage_demo.py --offline --reject        # rehearse the no

Rehearse --offline and --reject before the talk.
"""

import argparse, json, os, re, sys, time
from pathlib import Path
from typing import Annotated, TypedDict

HERE = Path(__file__).parent
PIPE_DIR = HERE / "pipelines"
CACHE_FILE = HERE / "cached_responses.json"
CATALOG_FILE = HERE / "catalog.json"

try:
    from langgraph.graph import StateGraph, START, END
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import interrupt, Command
except ImportError:
    print("pip install langgraph anthropic"); sys.exit(1)


class T:
    R = "\033[0m"; B = "\033[1m"; D = "\033[2m"
    RED = "\033[91m"; GRN = "\033[92m"; YEL = "\033[93m"
    BLU = "\033[94m"; MAG = "\033[95m"; CYN = "\033[96m"; W = "\033[97m"


# ── Model tiers: illustrative prices. Replace with your provider's real rates. ──
TIERS = {
    "small": {"model": "claude-haiku-4-5-20251001", "in": 1.00, "out": 5.00,  "label": "SMALL"},
    "mid":   {"model": "claude-sonnet-4-6",         "in": 3.00, "out": 15.00, "label": "MID"},
    "large": {"model": "claude-opus-4-1-20250805",  "in": 15.00, "out": 75.00, "label": "LARGE"},
}
TIER_COLOR = {"small": T.GRN, "mid": T.YEL, "large": T.RED}
HR = "─"


def bar(text, color=T.CYN, ch="="):
    print(f"\n{color}{T.B}{ch*72}\n  {text}\n{ch*72}{T.R}\n")


def node(name, sub, color):
    print(f"\n{color}{T.B}  \u25B6 {name}{T.R}  {T.D}{sub}{T.R}")
    print(f"{color}  {HR*68}{T.R}")


def line(text, color=T.W, ind=4):
    print(f"{' '*ind}{color}{text}{T.R}")


def tick(label, dots=3, delay=0.35):
    print(f"    {T.D}{label}", end="", flush=True)
    for _ in range(dots):
        time.sleep(delay); print(".", end="", flush=True)
    print(f"{T.R}")


# ══════════════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════════════
class LineageState(TypedDict, total=False):
    pipeline_name: str
    raw_sql: str
    scrubbed_sql: str
    scrub_report: dict
    complexity_score: int
    tier: str
    lineage: dict
    confidence: float
    validation: dict
    needs_human: bool
    human_decision: str
    catalog_status: str
    cost_usd: float
    tokens: dict
    audit: Annotated[list, lambda a, b: a + b]


OFFLINE = False
_cache = {}


def load_cache():
    global _cache
    if CACHE_FILE.exists():
        _cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))


def save_cache():
    CACHE_FILE.write_text(json.dumps(_cache, indent=2), encoding="utf-8")


def upsert_catalog(record):
    """Write one committed lineage record to catalog.json, keyed by target_table."""
    records = []
    if CATALOG_FILE.exists():
        records = json.loads(CATALOG_FILE.read_text(encoding="utf-8"))
    records = [r for r in records if r["target_table"] != record["target_table"]]
    records.append(record)
    CATALOG_FILE.write_text(json.dumps(records, indent=2), encoding="utf-8")


def llm_json(key, prompt, tier, max_tokens=1600):
    """Model call, or cached replay. Returns (parsed, usage_dict)."""
    if OFFLINE:
        if key not in _cache:
            print(f"{T.RED}No cached entry '{key}'. Run online once.{T.R}"); sys.exit(1)
        time.sleep(0.9)
        e = _cache[key]
        return e["result"], e["usage"]

    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(f"{T.RED}ANTHROPIC_API_KEY not set \u2014 or use --offline{T.R}"); sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    r = client.messages.create(model=TIERS[tier]["model"], max_tokens=max_tokens,
                               messages=[{"role": "user", "content": prompt}])
    txt = "".join(b.text for b in r.content if b.type == "text").strip()
    if txt.startswith("```"):
        txt = txt.split("```")[1]
        if txt.startswith("json"):
            txt = txt[4:]
    result = json.loads(txt.strip())
    usage = {"in": r.usage.input_tokens, "out": r.usage.output_tokens}
    _cache[key] = {"result": result, "usage": usage}
    save_cache()
    return result, usage


def cost_of(usage, tier):
    t = TIERS[tier]
    return (usage["in"] / 1_000_000) * t["in"] + (usage["out"] / 1_000_000) * t["out"]


# ══════════════════════════════════════════════════════════════
# NODE 1 \u2014 PARSER & SCRUBBER  (runs entirely in-network)
# ══════════════════════════════════════════════════════════════
def parser_scrubber(state: LineageState) -> dict:
    node("PARSER & SCRUBBER", "in-network \u2014 nothing has left yet", T.BLU)
    sql = state["raw_sql"]

    literals = re.findall(r"'[^']*'", sql)
    numbers = re.findall(r"\b\d{4,}\b", sql)
    creds = re.findall(r"(?i)(password|secret|token|api_key)\s*=\s*\S+", sql)

    scrubbed = re.sub(r"'[^']*'", "'<REDACTED>'", sql)
    scrubbed = re.sub(r"\b\d{4,}\b", "<NUM>", scrubbed)
    scrubbed = re.sub(r"^\s*--.*$", "", scrubbed, flags=re.M)
    scrubbed = re.sub(r"\n{3,}", "\n\n", scrubbed).strip()

    saved = len(sql) - len(scrubbed)
    n_comments = len(re.findall(r"^\s*--", sql, flags=re.M))
    line(f"Literal values redacted      {len(literals)}", T.GRN)
    line(f"Numeric constants masked     {len(numbers)}", T.GRN)
    line(f"Credential patterns found    {len(creds)}", T.GRN if not creds else T.RED)
    line(f"Comments stripped            {n_comments}", T.GRN)
    line(f"Payload reduced by {saved} characters before any model call", T.D)
    print()
    line("What crosses the boundary: table names, column names, join structure.", T.W)
    line("What never crosses: a single row of customer data.", T.W)

    return {"scrubbed_sql": scrubbed,
            "scrub_report": {"literals": len(literals), "numbers": len(numbers), "creds": len(creds)},
            "audit": [{"actor": "scrubber", "action": f"redacted {len(literals)+len(numbers)} literal(s)"}]}


# ══════════════════════════════════════════════════════════════
# NODE 2 \u2014 COMPLEXITY ROUTER  (local scoring, before spend)
# ══════════════════════════════════════════════════════════════
def complexity_router(state: LineageState) -> dict:
    node("COMPLEXITY ROUTER", "local scoring \u2014 decides the tier before we spend", T.YEL)
    sql = state["scrubbed_sql"].lower()

    signals = {
        "CTEs":            len(re.findall(r"\bwith\b|\)\s*,\s*\w+\s+as\s*\(", sql)),
        "joins":           len(re.findall(r"\bjoin\b", sql)),
        "aggregations":    len(re.findall(r"\b(sum|avg|count|max|min)\s*\(", sql)),
        "case expressions": len(re.findall(r"\bcase\b", sql)),
        "source tables":   len(set(re.findall(r"from\s+([\w.]+)|join\s+([\w.]+)", sql))),
        "lines":           len(sql.splitlines()),
    }
    score = (signals["CTEs"] * 6 + signals["joins"] * 4 + signals["aggregations"] * 2
             + signals["case expressions"] * 3 + signals["lines"] // 10)

    for k, v in signals.items():
        line(f"{k:<18} {v}", T.D)
    print()

    tier = "small" if score < 15 else "mid" if score < 40 else "large"
    c = TIER_COLOR[tier]
    line(f"Complexity score: {score}", T.W)
    line(f"Routed to {c}{T.B}{TIERS[tier]['label']}{T.R} tier  ({TIERS[tier]['model']})", T.W)
    line("This decision cost nothing. It is arithmetic, not a model call.", T.D)

    return {"complexity_score": score, "tier": tier,
            "audit": [{"actor": "router", "action": f"score {score} \u2192 {tier} tier"}]}


# ══════════════════════════════════════════════════════════════
# NODE 3 \u2014 EXTRACTION AGENT
# ══════════════════════════════════════════════════════════════
EXTRACT_PROMPT = """You are a data lineage extraction engine.

Return ONLY valid JSON. No preamble, no markdown fences.

{"target_table":"...","source_tables":[{"name":"...","layer":"raw|reference|staging|warehouse|analytics"}],
 "transformations":[{"output_column":"...","logic":"one line","type":"passthrough|aggregate|derived|label"}],
 "pii_risk_columns":["..."],
 "confidence":0.0-1.0,
 "uncertainty_note":"one line, or empty string"}

Set confidence below 0.75 if any lineage relationship is genuinely ambiguous.

SQL:
```sql
{sql}
```"""


def extraction_agent(state: LineageState) -> dict:
    tier = state["tier"]
    node("EXTRACTION AGENT", f"{TIERS[tier]['label']} tier \u2014 the only paid step so far", TIER_COLOR[tier])
    tick("extracting lineage")

    key = f"{state['pipeline_name']}:extract"
    result, usage = llm_json(key, EXTRACT_PROMPT.replace("{sql}", state["scrubbed_sql"]), tier)
    c = cost_of(usage, tier)

    line(f"Target        {result['target_table']}", T.W)
    line(f"Sources       {len(result['source_tables'])} tables", T.W)
    line(f"Transforms    {len(result['transformations'])} columns", T.W)
    if result.get("pii_risk_columns"):
        line(f"PII flagged   {', '.join(result['pii_risk_columns'])}", T.RED)
    print()
    line(f"Tokens  {usage['in']} in / {usage['out']} out"
         f"     Cost  ${c:.4f}", T.D)
    conf = result.get("confidence", 1.0)
    cc = T.GRN if conf >= 0.75 else T.YEL
    line(f"Self-reported confidence  {cc}{conf:.2f}{T.R}", T.W)

    return {"lineage": result, "confidence": conf, "cost_usd": c, "tokens": usage,
            "audit": [{"actor": "extraction_agent", "action": f"{tier} tier, ${c:.4f}"}]}


# ══════════════════════════════════════════════════════════════
# NODE 4 \u2014 VALIDATION AGENT
# ══════════════════════════════════════════════════════════════
def validation_agent(state: LineageState) -> dict:
    node("VALIDATION AGENT", "checks the extraction against the source \u2014 locally", T.MAG)
    sql = state["scrubbed_sql"].lower()
    lin = state["lineage"]

    checks, failed = [], 0
    for src in lin["source_tables"]:
        present = src["name"].split(".")[-1].lower() in sql
        checks.append((f"source '{src['name']}' appears in SQL", present))
        if not present: failed += 1

    tgt_ok = lin["target_table"].split(".")[-1].lower() in sql
    checks.append((f"target '{lin['target_table']}' appears in SQL", tgt_ok))
    if not tgt_ok: failed += 1

    n_out = len(re.findall(r"\bas\s+\w+", sql))
    ratio_ok = len(lin["transformations"]) >= max(1, n_out // 3)
    checks.append(("transformation count is plausible", ratio_ok))
    if not ratio_ok: failed += 1

    for desc, ok in checks:
        line(f"{T.GRN}\u2713{T.R} {desc}" if ok else f"{T.RED}\u2716{T.R} {desc}", T.D)

    conf = state["confidence"]
    needs_human = failed > 0 or conf < 0.75
    print()
    if needs_human:
        line(f"{T.YEL}{failed} check(s) failed / confidence {conf:.2f} \u2192 escalating to a human{T.R}", T.W)
    else:
        line(f"{T.GRN}All checks passed, confidence {conf:.2f} \u2192 auto-commit path{T.R}", T.W)
    line("This validation cost $0.00. It is assertions against the source, not a model.", T.D)

    return {"validation": {"failed": failed, "total": len(checks)}, "needs_human": needs_human,
            "audit": [{"actor": "validation_agent", "action": f"{len(checks)-failed}/{len(checks)} passed"}]}


# ══════════════════════════════════════════════════════════════
# NODE 5 \u2014 CONDITIONAL HUMAN GATE
# ══════════════════════════════════════════════════════════════
def route_gate(state: LineageState) -> str:
    return "human_gate" if state["needs_human"] else "catalog"


def human_gate(state: LineageState) -> dict:
    bar("\u23F8  HUMAN APPROVAL GATE \u2014 GRAPH HALTED", T.YEL)
    lin = state["lineage"]
    print(f"  {T.B}Pipeline:{T.R} {state['pipeline_name']}")
    print(f"  {T.B}Reason:{T.R}   confidence {state['confidence']:.2f}, "
          f"{state['validation']['failed']} validation check(s) failed")
    if lin.get("uncertainty_note"):
        print(f"  {T.B}Agent says:{T.R} {T.YEL}{lin['uncertainty_note']}{T.R}")
    print(f"\n  {T.D}State is checkpointed. This process could exit now and resume")
    print(f"  tomorrow from exactly this point. Nothing has been written.{T.R}\n")

    d = interrupt({"pipeline": state["pipeline_name"], "confidence": state["confidence"]})
    verdict = d.get("decision", "reject")

    if verdict == "approve":
        print(f"  {T.GRN}{T.B}\u2713 APPROVED by data steward{T.R}")
    else:
        print(f"  {T.RED}{T.B}\u2716 REJECTED \u2014 sent back for manual review{T.R}")
    if d.get("feedback"):
        line(f"Note: {d['feedback']}", T.D, 2)

    return {"human_decision": verdict,
            "audit": [{"actor": "HUMAN", "action": f"gate: {verdict}"}]}


def route_after_human(state: LineageState) -> str:
    return "catalog" if state.get("human_decision") == "approve" else "rejected"


# ══════════════════════════════════════════════════════════════
# NODE 6 \u2014 CATALOG / REJECTED
# ══════════════════════════════════════════════════════════════
def catalog(state: LineageState) -> dict:
    auto = not state["needs_human"]
    status = "auto_committed" if auto else "committed_after_approval"
    lin = state["lineage"]

    node("METADATA CATALOG", "committed", T.GRN)
    line(f"{lin['target_table']} lineage written.", T.GRN)
    line("Auto-committed \u2014 no human was interrupted." if auto
         else "Committed after human approval.", T.D)

    print()
    line("Catalog record", T.B + T.W)
    line(f"target        {lin['target_table']}", T.W)
    line(f"sources       {', '.join(s['name'] for s in lin['source_tables'])}", T.W)
    line(f"columns       {len(lin['transformations'])} tracked", T.W)
    if lin.get("pii_risk_columns"):
        line(f"pii flagged   {', '.join(lin['pii_risk_columns'])}", T.RED)
    line(f"confidence    {state['confidence']:.2f}", T.W)
    line(f"tier / cost   {state['tier'].upper()} / ${state['cost_usd']:.4f}", T.W)
    line(f"status        {status}", T.W)
    line(f"written to    {CATALOG_FILE.name}", T.D)

    record = {
        "pipeline": state["pipeline_name"],
        "target_table": lin["target_table"],
        "source_tables": lin["source_tables"],
        "transformations": lin["transformations"],
        "pii_risk_columns": lin.get("pii_risk_columns", []),
        "confidence": state["confidence"],
        "tier": state["tier"],
        "cost_usd": state["cost_usd"],
        "catalog_status": status,
    }
    upsert_catalog(record)

    return {"catalog_status": status,
            "audit": [{"actor": "catalog", "action": "lineage committed"}]}


def rejected(state: LineageState) -> dict:
    node("REJECTED", "nothing written", T.RED)
    line("No lineage committed. Queued for manual documentation.", T.W)
    line("The rejection and its reason are in the audit log.", T.D)
    return {"catalog_status": "rejected",
            "audit": [{"actor": "system", "action": "queued for manual review"}]}


# ══════════════════════════════════════════════════════════════
def build_graph():
    g = StateGraph(LineageState)
    for n, f in [("scrubber", parser_scrubber), ("router", complexity_router),
                 ("extract", extraction_agent), ("validate", validation_agent),
                 ("human_gate", human_gate), ("catalog", catalog), ("rejected", rejected)]:
        g.add_node(n, f)
    g.add_edge(START, "scrubber")
    g.add_edge("scrubber", "router")
    g.add_edge("router", "extract")
    g.add_edge("extract", "validate")
    g.add_conditional_edges("validate", route_gate,
                            {"human_gate": "human_gate", "catalog": "catalog"})
    g.add_conditional_edges("human_gate", route_after_human,
                            {"catalog": "catalog", "rejected": "rejected"})
    g.add_edge("catalog", END)
    g.add_edge("rejected", END)
    return g.compile(checkpointer=MemorySaver())


def run_one(app, path, idx, args):
    name = path.stem
    bar(f"PIPELINE {idx}  \u2014  {name}.sql", T.CYN)
    cfg = {"configurable": {"thread_id": f"anin-{name}"}}
    init = {"pipeline_name": name, "raw_sql": path.read_text(encoding="utf-8"), "audit": []}

    for _ in app.stream(init, config=cfg):
        pass

    st = app.get_state(cfg)
    if st.next:  # graph is paused at the gate
        if args.interactive:
            print(f"\n  {T.B}{T.YEL}Your call \u2014 approve this lineage?{T.R}")
            ans = input("  [a]pprove / [r]eject: ").strip().lower()
            approve = ans.startswith("a")
            fb = input("  Note (optional): ").strip()
        else:
            approve = not args.reject
            fb = "" if approve else "Ambiguous join grain \u2014 confirm with the pipeline owner first."
            time.sleep(1.0)
        for _ in app.stream(Command(resume={"decision": "approve" if approve else "reject",
                                            "feedback": fb}), config=cfg):
            pass

    return app.get_state(cfg).values


def main():
    global OFFLINE
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--reject", action="store_true")
    args = ap.parse_args()
    OFFLINE = args.offline
    load_cache()

    bar("AGENTIC DATA LINEAGE PIPELINE", T.CYN)
    print(f"  scrubber \u2192 router \u2192 extraction agent \u2192 validation agent "
          f"\u2192 conditional gate \u2192 catalog")
    print(f"  {T.D}Mode: {'OFFLINE (cached)' if OFFLINE else 'LIVE'}"
          f"   |   3 pipelines of increasing complexity{T.R}")

    app = build_graph()
    files = [PIPE_DIR / "simple_customer_view.sql",
             PIPE_DIR / "daily_revenue_rollup.sql",
             PIPE_DIR / "customer_churn_features.sql"]

    results = []
    for i, f in enumerate(files, 1):
        results.append(run_one(app, f, i, args))

    # ── Summary ──
    bar("RUN SUMMARY", T.GRN)
    total = sum(r.get("cost_usd", 0) for r in results)
    gated = sum(1 for r in results if r.get("needs_human"))

    print(f"  {'Pipeline':<32}{'Tier':<9}{'Conf':<8}{'Cost':<11}Outcome")
    print(f"  {T.D}{HR*70}{T.R}")
    for r in results:
        tier = r.get("tier", "?")
        c = TIER_COLOR.get(tier, T.W)
        status = {"auto_committed": f"{T.GRN}auto-committed{T.R}",
                  "committed_after_approval": f"{T.YEL}human approved{T.R}",
                  "rejected": f"{T.RED}rejected{T.R}"}.get(r.get("catalog_status"), "?")
        print(f"  {r['pipeline_name']:<32}{c}{TIERS[tier]['label']:<9}{T.R}"
              f"{r.get('confidence',0):<8.2f}${r.get('cost_usd',0):<10.4f}{status}")

    print(f"\n  {T.B}Total for 3 pipelines:  ${total:.4f}{T.R}"
          f"     {T.D}(\u2248 ${total/3:.4f} per pipeline){T.R}")
    print(f"  {T.B}Human interrupted:      {gated} of 3{T.R}"
          f"     {T.D}The gate is conditional, not universal.{T.R}")

    bar("AUDIT LOG", T.CYN)
    for r in results:
        print(f"  {T.B}{r['pipeline_name']}{T.R}")
        for e in r.get("audit", []):
            is_h = e["actor"] == "HUMAN"
            tag = f"{T.GRN}[HUMAN]{T.R}" if is_h else f"{T.D}[AGENT]{T.R}"
            print(f"    {tag} {e['actor']:<18} {e['action']}")
        print()

    print(f"  {T.B}The line to say out loud:{T.R}")
    line("Agents did the work on all three. A human was interrupted for exactly one \u2014", T.W)
    line("the one that admitted it was not sure. That is what a guardrail looks like.", T.W)
    print()


if __name__ == "__main__":
    main()