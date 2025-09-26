from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import orjson
import pandas as pd

from .config import PATHS
from .mapping import attach_controls


@dataclass
class Policy:
    threshold: float
    cost_false_negative: float
    cost_false_positive: float
    impact_default: float

    def to_json(self) -> str:
        return json.dumps(
            {
                "threshold": self.threshold,
                "cost_false_negative": self.cost_false_negative,
                "cost_false_positive": self.cost_false_positive,
                "impact_default": self.impact_default,
            },
            indent=2,
        )


def compute_register(
    df: pd.DataFrame, policy: Policy, impact_column: Optional[str] = None
) -> pd.DataFrame:
    df = attach_controls(df)
    impact = (
        df[impact_column]
        if impact_column and impact_column in df.columns
        else policy.impact_default
    )
    df = df.copy()
    df["susceptibility"] = df["probability"].clip(0.0, 1.0)
    df["susceptibility_bucket"] = pd.cut(
        df["susceptibility"],
        bins=[0.0, 0.33, 0.66, 1.0],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )
    df["expected_loss"] = df["susceptibility"] * float(impact)
    return df


def write_register(df: pd.DataFrame, name: str):
    PATHS.register_dir.mkdir(parents=True, exist_ok=True)
    PATHS.policies_dir.mkdir(parents=True, exist_ok=True)
    csv_path = PATHS.register_dir / f"{name}.csv"
    json_path = PATHS.register_dir / f"{name}.json"
    df.to_csv(csv_path, index=False)
    with open(json_path, "wb") as f:
        f.write(orjson.dumps(df.to_dict(orient="records")))
