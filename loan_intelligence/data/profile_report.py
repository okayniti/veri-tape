"""Data quality profiling report for the raw (messy) loan tape.

Deliberately does NOT coerce/clean anything -- it profiles loans.csv and
payments.csv exactly as delivered, so the report reflects the real
messiness a reviewer would see before any feature engineering happens.
Output is a single self-contained HTML file (no external assets) plus a
markdown companion for quick terminal/CI reading.

Run directly: `python -m loan_intelligence.data.profile_report`
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_LOANS = BASE_DIR / "outputs" / "loans.csv"
DEFAULT_PAYMENTS = BASE_DIR / "outputs" / "payments.csv"
DEFAULT_REPORT_DIR = BASE_DIR / "reports"


def _numeric_like(series: pd.Series) -> pd.Series:
    """Best-effort numeric coercion used only to *detect* non-numeric junk,
    not to clean the data -- strips a trailing '%' before parsing so a
    format inconsistency doesn't register as 100% unparseable."""
    return pd.to_numeric(series.astype(str).str.rstrip("%"), errors="coerce")


def _profile_column(df: pd.DataFrame, col: str) -> dict:
    s = df[col]
    n = len(s)
    missing = s.isna().sum()
    info: dict = {
        "column": col,
        "dtype": str(s.dtype),
        "missing_count": int(missing),
        "missing_pct": round(100 * missing / n, 2) if n else 0.0,
        "n_unique": int(s.nunique(dropna=True)),
    }

    if s.dtype == object:
        coerced = _numeric_like(s)
        unparseable = coerced.isna().sum() - missing
        info["looks_numeric_but_object"] = bool(unparseable < 0.5 * n and coerced.notna().sum() > 0)
        info["unparseable_non_null_count"] = int(max(unparseable, 0))
        top = s.value_counts(dropna=True).head(5)
        info["top_values"] = "; ".join(f"{k!r}: {v}" for k, v in top.items())
        info["cardinality_flag"] = "high" if info["n_unique"] > 0.5 * n else ("low" if info["n_unique"] <= 20 else "normal")
    else:
        desc = s.describe()
        info["min"] = float(desc.get("min", np.nan))
        info["max"] = float(desc.get("max", np.nan))
        info["mean"] = round(float(desc.get("mean", np.nan)), 4) if n else None
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = s[(s < lo) | (s > hi)].dropna()
        info["iqr_outlier_count"] = int(len(outliers))
        info["iqr_outlier_pct"] = round(100 * len(outliers) / n, 2) if n else 0.0
        info["iqr_bounds"] = f"[{lo:.3f}, {hi:.3f}]"

    return info


KNOWN_RANGES = {
    "credit_score_at_origination": (300, 850),
    "interest_rate": (0, 40),
    "dti_at_origination": (0, 1.0),
}


def _domain_violations(df: pd.DataFrame) -> list[dict]:
    violations = []
    for col, (lo, hi) in KNOWN_RANGES.items():
        if col not in df.columns:
            continue
        numeric = _numeric_like(df[col])
        bad = numeric[(numeric.notna()) & ((numeric < lo) | (numeric > hi))]
        if len(bad):
            violations.append(
                {
                    "column": col,
                    "expected_range": f"[{lo}, {hi}]",
                    "violation_count": int(len(bad)),
                    "example_values": bad.head(5).tolist(),
                }
            )
    return violations


def _duplicate_summary(df: pd.DataFrame, key: str) -> dict:
    dupe_mask = df[key].duplicated(keep=False)
    return {
        "duplicate_key_rows": int(dupe_mask.sum()),
        "duplicate_key_values": int(df.loc[dupe_mask, key].nunique()),
    }


def _categorical_variant_summary(df: pd.DataFrame, cols: list[str]) -> list[dict]:
    """Flags likely-same-category spelling/casing variants (e.g. 'West' vs
    'WEST' vs 'w') by grouping on a normalized (lowercased, first-letter)
    key -- a cheap heuristic, not a general dedup, but enough to surface the
    cardinality inflation this dataset injects on purpose."""
    out = []
    for col in cols:
        if col not in df.columns:
            continue
        s = df[col].dropna().astype(str)
        norm = s.str.lower().str.replace("_loan", "", regex=False).str[:1]
        groups = s.groupby(norm).nunique()
        inflated = groups[groups > 1]
        if len(inflated):
            variants = {}
            for key_ in inflated.index:
                variants[key_] = sorted(s[norm == key_].unique().tolist())
            out.append({"column": col, "inflated_groups": variants})
    return out


def profile_dataset(loans: pd.DataFrame, payments: pd.DataFrame) -> dict:
    loans_profile = [_profile_column(loans, c) for c in loans.columns]
    payments_profile = [_profile_column(payments, c) for c in payments.columns]

    return {
        "loans_shape": loans.shape,
        "payments_shape": payments.shape,
        "loans_columns": loans_profile,
        "payments_columns": payments_profile,
        "loans_domain_violations": _domain_violations(loans),
        "loans_duplicates": _duplicate_summary(loans, "loan_id"),
        "categorical_variants": _categorical_variant_summary(loans, ["region", "loan_type"]),
    }


def _fmt_col_row(info: dict) -> str:
    extra_bits = []
    if "min" in info:
        extra_bits.append(f"range [{info['min']:.3g}, {info['max']:.3g}], mean {info['mean']}")
        extra_bits.append(f"IQR outliers: {info['iqr_outlier_count']} ({info['iqr_outlier_pct']}%), bounds {info['iqr_bounds']}")
    else:
        extra_bits.append(f"top values: {info.get('top_values', '')}")
        if info.get("looks_numeric_but_object"):
            extra_bits.append(f"⚠ looks numeric but stored as text ({info['unparseable_non_null_count']} unparseable)")
        extra_bits.append(f"cardinality: {info.get('cardinality_flag')}")
    return (
        f"<tr><td>{info['column']}</td><td>{info['dtype']}</td>"
        f"<td>{info['missing_count']} ({info['missing_pct']}%)</td>"
        f"<td>{info['n_unique']}</td><td>{'<br>'.join(extra_bits)}</td></tr>"
    )


def render_html(profile: dict) -> str:
    loans_rows = "\n".join(_fmt_col_row(c) for c in profile["loans_columns"])
    payments_rows = "\n".join(_fmt_col_row(c) for c in profile["payments_columns"])

    violations_rows = "".join(
        f"<tr><td>{v['column']}</td><td>{v['expected_range']}</td><td>{v['violation_count']}</td>"
        f"<td>{v['example_values']}</td></tr>"
        for v in profile["loans_domain_violations"]
    ) or "<tr><td colspan='4'>none detected</td></tr>"

    variants_html = ""
    for entry in profile["categorical_variants"]:
        rows = "".join(f"<li><code>{k}</code> → {v}</li>" for k, v in entry["inflated_groups"].items())
        variants_html += f"<h3>{entry['column']}</h3><ul>{rows}</ul>"
    variants_html = variants_html or "<p>none detected</p>"

    dupes = profile["loans_duplicates"]

    return f"""
<title>Loan Tape Profiling Report</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, sans-serif; margin: 2rem; color: #1a1a1a; background:#fafafa; }}
  h1, h2 {{ border-bottom: 2px solid #ddd; padding-bottom: 0.3rem; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; background: #fff; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 10px; font-size: 0.85rem; text-align: left; vertical-align: top; }}
  th {{ background: #f0f0f0; }}
  .summary {{ display:flex; gap:2rem; margin-bottom:1.5rem; }}
  .card {{ background:#fff; border:1px solid #ddd; border-radius:8px; padding:1rem 1.5rem; }}
  .card b {{ font-size:1.4rem; display:block; }}
  code {{ background:#eee; padding:1px 4px; border-radius:3px; }}
</style>
<h1>Loan Tape Profiling Report</h1>
<p>Profiled directly from the delivered (uncleaned) CSVs — reflects real data-quality issues before any feature engineering.</p>

<div class="summary">
  <div class="card"><b>{profile['loans_shape'][0]:,}</b>loan records</div>
  <div class="card"><b>{profile['loans_shape'][1]}</b>loan columns</div>
  <div class="card"><b>{profile['payments_shape'][0]:,}</b>payment-month rows</div>
  <div class="card"><b>{dupes['duplicate_key_rows']}</b>rows with a duplicated loan_id ({dupes['duplicate_key_values']} distinct ids)</div>
</div>

<h2>loans.csv — column profile</h2>
<table>
<tr><th>column</th><th>dtype</th><th>missing</th><th>n_unique</th><th>notes</th></tr>
{loans_rows}
</table>

<h2>payments.csv — column profile</h2>
<table>
<tr><th>column</th><th>dtype</th><th>missing</th><th>n_unique</th><th>notes</th></tr>
{payments_rows}
</table>

<h2>Domain-range violations</h2>
<table>
<tr><th>column</th><th>expected range</th><th>violation count</th><th>examples</th></tr>
{violations_rows}
</table>

<h2>Categorical spelling/casing variants</h2>
{variants_html}
"""


def render_markdown(profile: dict) -> str:
    lines = ["# Loan Tape Profiling Report", ""]
    lines.append(f"- loans.csv: {profile['loans_shape'][0]:,} rows x {profile['loans_shape'][1]} cols")
    lines.append(f"- payments.csv: {profile['payments_shape'][0]:,} rows x {profile['payments_shape'][1]} cols")
    d = profile["loans_duplicates"]
    lines.append(f"- duplicated loan_id rows: {d['duplicate_key_rows']} ({d['duplicate_key_values']} distinct ids)")
    lines.append("")
    lines.append("## loans.csv columns")
    lines.append("| column | dtype | missing | n_unique |")
    lines.append("|---|---|---|---|")
    for c in profile["loans_columns"]:
        lines.append(f"| {c['column']} | {c['dtype']} | {c['missing_count']} ({c['missing_pct']}%) | {c['n_unique']} |")
    lines.append("")
    lines.append("## Domain-range violations")
    if profile["loans_domain_violations"]:
        for v in profile["loans_domain_violations"]:
            lines.append(f"- **{v['column']}** expected {v['expected_range']}: {v['violation_count']} violations, examples {v['example_values']}")
    else:
        lines.append("- none detected")
    lines.append("")
    lines.append("## Categorical variants")
    if profile["categorical_variants"]:
        for entry in profile["categorical_variants"]:
            lines.append(f"- **{entry['column']}**: {entry['inflated_groups']}")
    else:
        lines.append("- none detected")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile the raw (messy) loan tape.")
    parser.add_argument("--loans", type=Path, default=DEFAULT_LOANS)
    parser.add_argument("--payments", type=Path, default=DEFAULT_PAYMENTS)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()

    loans = pd.read_csv(args.loans)
    payments = pd.read_csv(args.payments)
    profile = profile_dataset(loans, payments)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    html_path = args.report_dir / "profile_report.html"
    md_path = args.report_dir / "profile_report.md"
    html_path.write_text(render_html(profile), encoding="utf-8")
    md_path.write_text(render_markdown(profile), encoding="utf-8")

    print(f"wrote {html_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
