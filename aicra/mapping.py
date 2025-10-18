from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import get_settings


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def normalize_family(family: str, mapping: dict[str, Any]) -> str:
    norm = family
    if mapping.get("normalize", {}).get("lowercase", True):
        norm = norm.lower()
    if mapping.get("normalize", {}).get("strip", True):
        norm = norm.strip()
    repl = mapping.get("normalize", {}).get("replace", {})
    for k, v in repl.items():
        norm = norm.replace(k, v)
    return norm


def family_to_label(family: str, mapping: dict[str, Any]) -> int:
    norm = normalize_family(family, mapping)
    ransomware_fams = set(mapping.get("families", {}).get("ransomware", []))
    return int(norm in ransomware_fams)


def families_to_attack(family: str, attack_map: dict[str, Any]) -> list[str]:
    return attack_map.get(family.lower(), [])


def attack_to_d3fend(techniques: list[str], d3f: dict[str, Any]) -> list[dict[str, Any]]:
    controls: list[dict[str, Any]] = []
    for t in techniques:
        controls.extend(d3f.get(t, []))
    return controls


def attach_controls(df: pd.DataFrame) -> pd.DataFrame:
    settings = get_settings()
    fam_map = _load_json(settings.mappings_dir / "family_mapping.json")
    attack_map = _load_json(settings.mappings_dir / "attack_mapping.json")
    d3f = _load_json(settings.mappings_dir / "d3fend_graph.json")
    techniques = df["family"].apply(
        lambda f: families_to_attack(normalize_family(f, fam_map), attack_map)
    )
    controls = techniques.apply(lambda ts: attack_to_d3fend(ts, d3f))
    df = df.copy()
    df["attack_techniques"] = techniques
    df["d3fend_controls"] = controls
    return df
