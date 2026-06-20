#!/usr/bin/env python3
"""Generate side-by-side deterministic vs learned mapping examples for H3."""

from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd

MITRE_D3FEND_CSV_URL = (
    "https://raw.githubusercontent.com/mitre/d3fend/master/ontologies/d3fend.csv"
)


def _parse_yaml_control_comments(yaml_path: Path) -> dict[str, str]:
    """Parse inline comments from attack_to_d3fend.yaml (e.g. D3-BDR # Backup Data Recovery)."""
    if not yaml_path.exists():
        return {}

    legend: dict[str, str] = {}
    pattern = re.compile(r"^\s*-\s*(D3-[A-Z0-9]+)\s*#\s*(.+?)\s*$")
    for line in yaml_path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            legend[match.group(1)] = match.group(2).strip()
    return legend


def _load_mitre_d3fend_legend(cache_path: Path) -> dict[str, str]:
    """Load MITRE D3FEND control names (cached locally after first fetch)."""
    if cache_path.exists():
        df = pd.read_csv(cache_path)
    else:
        try:
            import requests

            response = requests.get(MITRE_D3FEND_CSV_URL, timeout=30)
            response.raise_for_status()
            df = pd.read_csv(io.StringIO(response.text))
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_path, index=False)
        except Exception:
            return {}

    legend: dict[str, str] = {}
    for _, row in df.iterrows():
        control_id = str(row.get("ID", "")).strip()
        if not control_id.startswith("D3-"):
            continue

        name = row.get("D3FEND Technique")
        if pd.isna(name) or not str(name).strip():
            name = row.get("D3FEND Technique Level 0")
        if pd.isna(name) or not str(name).strip():
            name = row.get("D3FEND Technique Level 1")
        if pd.isna(name) or not str(name).strip():
            continue

        legend[control_id] = str(name).strip()
    return legend


def load_control_legend(root: Path) -> dict[str, str]:
    """Merge control ID -> human-readable name from all available local/remote sources."""
    legend: dict[str, str] = {}

    cache_path = (
        root / "results" / "H3_full_evaluation" / "d3fend_control_legend_cache.csv"
    )
    legend.update(_load_mitre_d3fend_legend(cache_path))

    det_path = root / "data" / "mappings" / "deterministic_attack_defense_lookup.csv"
    if det_path.exists():
        det = pd.read_csv(det_path)
        if {"defense_id", "defense_name"}.issubset(det.columns):
            for _, row in (
                det[["defense_id", "defense_name"]].drop_duplicates().iterrows()
            ):
                legend[str(row["defense_id"])] = str(row["defense_name"])

    legend.update(
        _parse_yaml_control_comments(
            root / "data" / "lookups" / "attack_to_d3fend.yaml"
        )
    )

    register_path = (
        root
        / "register"
        / "h1h2_rebuild"
        / "small_ember"
        / "ransomware_only_risk_register.csv"
    )
    if register_path.exists():
        reg = pd.read_csv(
            register_path, usecols=["d3fend_control_id", "d3fend_control_name"]
        )
        for _, row in reg.drop_duplicates().iterrows():
            legend[str(row["d3fend_control_id"])] = str(row["d3fend_control_name"])

    return legend


def format_controls_labeled(controls: list[str], legend: dict[str, str]) -> str:
    """Format controls as 'D3-RA (Restore Access); ...'."""
    parts = []
    for control_id in controls:
        name = legend.get(control_id)
        if name:
            parts.append(f"{control_id} ({name})")
        else:
            parts.append(control_id)
    return "; ".join(parts)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    det_path = root / "data" / "mappings" / "deterministic_attack_defense_lookup.csv"
    lrn_path = root / "data" / "mappings" / "learned_mapping.csv"
    out_md = root / "results" / "H3_full_evaluation" / "mapping_comparison_examples.md"
    out_csv = (
        root / "results" / "H3_full_evaluation" / "mapping_comparison_examples.csv"
    )
    out_csv_labeled = (
        root
        / "results"
        / "H3_full_evaluation"
        / "mapping_comparison_examples_labeled.csv"
    )
    out_legend_csv = (
        root / "results" / "H3_full_evaluation" / "mapping_control_legend.csv"
    )

    control_legend = load_control_legend(root)

    det = pd.read_csv(det_path)
    lrn = pd.read_csv(lrn_path)

    if "attack_id" in det.columns:
        det = det.rename(
            columns={"attack_id": "technique_id", "defense_id": "control_id"}
        )
    if "is_correct" in det.columns:
        det = det[det["is_correct"] == 1]

    names: dict[str, str] = {}
    if "attack_name" in det.columns:
        for _, row in det.drop_duplicates("technique_id").iterrows():
            names[str(row["technique_id"])] = str(row.get("attack_name", ""))

    def agg_controls(df: pd.DataFrame) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {}
        for tid, g in df.groupby("technique_id"):
            grouped[str(tid)] = sorted(g["control_id"].astype(str).unique().tolist())
        return grouped

    det_map = agg_controls(det)
    lrn_map = agg_controls(lrn)

    shared_techniques = sorted(set(det_map) & set(lrn_map))
    n_examples = 15

    priority = [
        "T1486",
        "T1490",
        "T1485",
        "T1487",
        "T1488",
        "T1055",
        "T1070",
        "T1021",
        "T1055.011",
        "T1070.006",
        "T1485.001",
        "T1489",
        "T1021.005",
        "T1070.002",
        "T1041",
    ]
    selected: list[str] = []
    for tid in priority:
        if tid in shared_techniques:
            selected.append(tid)
        if len(selected) >= n_examples:
            break

    if len(selected) < n_examples:
        for tid in shared_techniques:
            if tid not in selected:
                selected.append(tid)
            if len(selected) >= n_examples:
                break

    rows = []
    used_controls: set[str] = set()
    for tid in selected:
        d = det_map.get(tid, [])
        lrn_controls = lrn_map.get(tid, [])
        used_controls.update(d)
        used_controls.update(lrn_controls)
        overlap = sorted(set(d) & set(lrn_controls))
        if set(d) == set(lrn_controls) and d:
            match = "EXACT"
        elif overlap:
            match = "PARTIAL"
        else:
            match = "DISJOINT"

        rows.append(
            {
                "technique_id": tid,
                "technique_name": names.get(tid, "—"),
                "det_count": len(d),
                "lrn_count": len(lrn_controls),
                "deterministic_controls_full": "; ".join(d),
                "learned_controls_full": "; ".join(lrn_controls),
                "deterministic_controls_labeled": format_controls_labeled(
                    d, control_legend
                ),
                "learned_controls_labeled": format_controls_labeled(
                    lrn_controls, control_legend
                ),
                "deterministic_controls": ", ".join(d[:6])
                + ("…" if len(d) > 6 else ""),
                "learned_controls": ", ".join(lrn_controls[:6])
                + ("…" if len(lrn_controls) > 6 else ""),
                "overlap": "; ".join(overlap) if overlap else "",
                "match": match,
            }
        )

    n_rows = len(rows)
    lines = [
        f"# H3 Mapping Comparison: Deterministic vs Learned ({n_rows} Examples)",
        "",
        "**Selection rule:** Only ATT&CK techniques mapped in **both** deterministic and "
        "learned files (both columns populated).",
        "",
        "**Source files (H3 experiment):**",
        f"- Deterministic: `data/mappings/deterministic_attack_defense_lookup.csv` "
        f"(ransomware-focused, {len(det)} pairs, {len(det_map)} techniques)",
        f"- Learned: `data/mappings/learned_mapping.csv` "
        f"({len(lrn)} pairs, {len(lrn_map)} techniques)",
        f"- Shared techniques available: {len(shared_techniques)}",
        "",
        "## D3FEND control legend (symbols used in this table)",
        "",
        "Each control ID is a MITRE D3FEND countermeasure shorthand. "
        "Full legend for controls in this table:",
        "",
        "| Control ID | Meaning |",
        "|------------|---------|",
    ]

    for control_id in sorted(used_controls):
        meaning = control_legend.get(
            control_id, "Name not found in local/MITRE catalog"
        )
        lines.append(f"| {control_id} | {meaning} |")

    lines.extend(
        [
            "",
            "See also: `mapping_control_legend.csv` and "
            "`mapping_comparison_examples_labeled.csv` (easy to copy into Excel).",
            "",
            "| # | ATT&CK ID | Technique | Det # | Lrn # | "
            "Deterministic controls | Learned controls | Overlap | Match |",
            "|---|-----------|-----------|------:|------:|"
            "------------------------|-------------------|---------|-------|",
        ]
    )

    for i, row in enumerate(rows, 1):
        lines.append(
            f"| {i} | {row['technique_id']} | {row['technique_name']} | "
            f"{row['det_count']} | {row['lrn_count']} | "
            f"{row['deterministic_controls']} | {row['learned_controls']} | "
            f"{row['overlap'] or '—'} | {row['match']} |"
        )

    lines.extend(
        [
            "",
            "## Side-by-side with control meanings (copy-friendly)",
            "",
            "| # | ATT&CK ID | Deterministic (with meanings) | Learned (with meanings) | Match |",
            "|---|-----------|--------------------------------|-------------------------|-------|",
        ]
    )

    for i, row in enumerate(rows, 1):
        lines.append(
            f"| {i} | {row['technique_id']} | {row['deterministic_controls_labeled']} | "
            f"{row['learned_controls_labeled']} | {row['match']} |"
        )

    exact = sum(1 for r in rows if r["match"] == "EXACT")
    partial = sum(1 for r in rows if r["match"] == "PARTIAL")
    disjoint = sum(1 for r in rows if r["match"] == "DISJOINT")

    lines.extend(
        [
            "",
            "## Key takeaway",
            "",
            f"- EXACT match: {exact}/{n_rows}",
            f"- PARTIAL overlap: {partial}/{n_rows}",
            f"- DISJOINT (no shared controls): {disjoint}/{n_rows}",
            "",
            "Deterministic mappings prioritize ransomware-relevant D3FEND controls "
            "(e.g., D3-RA Restore Access). Learned mappings assign broader heuristic "
            "controls with little or no overlap — consistent with H3 DAC findings "
            "(deterministic 100%, learned 0%).",
        ]
    )

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")

    csv_df = pd.DataFrame(
        [
            {
                "example_num": i,
                "technique_id": row["technique_id"],
                "technique_name": row["technique_name"],
                "det_count": row["det_count"],
                "lrn_count": row["lrn_count"],
                "deterministic_controls": row["deterministic_controls_full"],
                "learned_controls": row["learned_controls_full"],
                "deterministic_controls_explained": row[
                    "deterministic_controls_labeled"
                ],
                "learned_controls_explained": row["learned_controls_labeled"],
                "overlap": row["overlap"],
                "match": row["match"],
            }
            for i, row in enumerate(rows, 1)
        ]
    )
    csv_df.to_csv(out_csv_labeled, index=False, encoding="utf-8")
    try:
        csv_df.to_csv(out_csv, index=False, encoding="utf-8")
    except PermissionError:
        print(f"Warning: could not overwrite {out_csv} (file may be open).")
        print(f"Wrote labeled copy to {out_csv_labeled} instead.")

    legend_df = pd.DataFrame(
        [
            {
                "control_id": control_id,
                "control_name": control_legend.get(control_id, ""),
                "used_in_comparison_table": control_id in used_controls,
            }
            for control_id in sorted(control_legend)
        ]
    )
    legend_df.to_csv(out_legend_csv, index=False, encoding="utf-8")

    print(f"Wrote {out_md}")
    print(f"Wrote {out_csv_labeled}")
    if out_csv.exists():
        print(f"Updated {out_csv}")
    print(f"Wrote {out_legend_csv}")


if __name__ == "__main__":
    main()
