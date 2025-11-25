import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Any

import pandas as pd
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from src.utils.config import load_config

CFG_PATH = "config/dqi_config.yaml"


def load_dqi_config():
    return load_config(CFG_PATH)


@dataclass
class DriftMetrics:
    baseline_rows: int
    current_rows: int
    volume_change_pct: float
    column_null_change_pct: Dict[str, float]
    numeric_mean_change_pct: Dict[str, float]
    categorical_dist_change: Dict[str, Dict[str, float]]


# ---------- Ingress Agent ----------

def ingress_agent(cfg) -> Dict[str, pd.DataFrame]:
    """Loads baseline and current claim batches."""
    base = pd.read_csv(cfg.paths.baseline_claims)
    curr = pd.read_csv(cfg.paths.current_claims)
    return {"baseline": base, "current": curr}


# ---------- Profiler Agent ----------

def profiler_agent(cfg, dfs: Dict[str, pd.DataFrame]) -> DriftMetrics:
    baseline = dfs["baseline"]
    current = dfs["current"]

    base_rows = len(baseline)
    curr_rows = len(current)
    volume_change = (curr_rows - base_rows) / base_rows if base_rows else 0.0

    # Null drift per column
    null_change: Dict[str, float] = {}
    for col in baseline.columns:
        base_null = baseline[col].isna().mean()
        curr_null = current[col].isna().mean()
        null_change[col] = curr_null - base_null

    # Numeric mean drift
    num_mean_change: Dict[str, float] = {}
    for col in cfg.columns.numeric:
        if col in baseline.columns:
            base_mean = baseline[col].mean()
            curr_mean = current[col].mean()
            if base_mean:
                num_mean_change[col] = (curr_mean - base_mean) / base_mean
            else:
                num_mean_change[col] = 0.0

    # Categorical distribution drift
    cat_change: Dict[str, Dict[str, float]] = {}
    for col in cfg.columns.categorical:
        if col not in baseline.columns:
            continue
        base_dist = baseline[col].value_counts(normalize=True)
        curr_dist = current[col].value_counts(normalize=True)
        keys = set(base_dist.index) | set(curr_dist.index)
        col_delta: Dict[str, float] = {}
        for k in keys:
            col_delta[str(k)] = float(curr_dist.get(k, 0.0) - base_dist.get(k, 0.0))
        cat_change[col] = col_delta

    return DriftMetrics(
        baseline_rows=base_rows,
        current_rows=curr_rows,
        volume_change_pct=volume_change,
        column_null_change_pct=null_change,
        numeric_mean_change_pct=num_mean_change,
        categorical_dist_change=cat_change,
    )


# ---------- Anomaly Reasoning Agent (LLM via Ollama) ----------

def anomaly_agent(cfg, metrics: DriftMetrics) -> Dict[str, Any]:
    llm = ChatOllama(
        model=cfg.llm.model,
        temperature=cfg.llm.temperature,
    )

    prompt = f"""
You are a senior healthcare data quality engineer working on Medicaid/Medicare claims.

You are given data drift metrics between a baseline week and a current week.

Metrics JSON:
{json.dumps(asdict(metrics), indent=2)}

Thresholds JSON:
{json.dumps(cfg.drift_thresholds.__dict__, indent=2) if hasattr(cfg, 'drift_thresholds') else '{}'}

Decide:
1. Is there data drift? What kind (volume, nulls, amounts, status mix, CPT mix)?
2. What are the likely root causes? (ETL issue, upstream system change, real business event, potential fraud spike, etc.)
3. How severe is it overall? (LOW, MEDIUM, HIGH)
4. What specific checks or SQL queries should an engineer run next?

Respond in VALID JSON only, with keys:
- "summary"
- "severity"
- "suspected_causes" (list of strings)
- "recommended_checks" (list of strings)
- "business_risk" (string)
"""

    resp = llm.invoke([HumanMessage(content=prompt)])
    text = resp.content

    # Try to extract JSON even if the model wraps it in text/backticks
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        json_str = text[start:end]
        parsed = json.loads(json_str)
    except Exception:
        parsed = {"raw_response": text, "severity": "UNKNOWN"}

    return parsed


# ---------- Reporter Agent ----------

def reporter_agent(cfg, metrics: DriftMetrics, analysis: Dict[str, Any]) -> None:
    md_path = cfg.paths.report_md
    json_path = cfg.paths.report_json

    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    # JSON payload
    payload = {
        "metrics": asdict(metrics),
        "analysis": analysis,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    # Markdown report
    lines = []
    lines.append("# Agentic Data Quality Investigator Report\n")
    lines.append(f"Project: **{cfg.project_name}**\n")

    lines.append("## Volume\n")
    lines.append(f"- Baseline rows: `{metrics.baseline_rows}`")
    lines.append(f"- Current rows: `{metrics.current_rows}`")
    lines.append(f"- Volume change: `{metrics.volume_change_pct:.2%}`\n")

    lines.append("## Numeric Mean Drift\n")
    for col, delta in metrics.numeric_mean_change_pct.items():
        lines.append(f"- **{col}** mean change: `{delta:.2%}`")
    lines.append("")

    lines.append("## Null Drift (increase in null ratio)\n")
    for col, delta in metrics.column_null_change_pct.items():
        if delta > 0:
            lines.append(f"- **{col}** nulls increased by `{delta:.2%}`")
    lines.append("")

    lines.append("## Categorical Distribution Changes\n")
    for col, deltas in metrics.categorical_dist_change.items():
        lines.append(f"- **{col}**:")
        for val, d in deltas.items():
            if abs(d) > 0.05:
                sign = "↑" if d > 0 else "↓"
                lines.append(f"  - `{val}` {sign} `{d:.2%}`")
    lines.append("\n---\n")

    lines.append("## LLM Analysis\n")
    lines.append(f"**Severity:** `{analysis.get('severity', 'UNKNOWN')}`\n")
    lines.append(f"**Summary:** {analysis.get('summary', '')}\n")

    if "suspected_causes" in analysis:
        lines.append("\n**Suspected Causes:**")
        for c in analysis["suspected_causes"]:
            lines.append(f"- {c}")

    if "recommended_checks" in analysis:
        lines.append("\n**Recommended Checks:**")
        for c in analysis["recommended_checks"]:
            lines.append(f"- {c}")

    lines.append("\n**Business Risk:**")
    lines.append(analysis.get("business_risk", analysis.get("summary", "")))

    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Markdown report written to {md_path}")
    print(f"JSON report written to {json_path}")
