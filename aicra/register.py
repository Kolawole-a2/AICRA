from __future__ import annotations

import json
from dataclasses import dataclass

import orjson
import pandas as pd

from .config import get_settings
from .pipelines.mapping import MappingPipeline


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
    df: pd.DataFrame, policy: Policy, impact_column: str | None = None
) -> pd.DataFrame:
    settings = get_settings()
    mapping_pipeline = MappingPipeline(settings)
    
    # Enrich with mapped controls
    df = df.copy()
    
    # Apply mappings
    mapping_results = df["family"].apply(mapping_pipeline.get_complete_mapping)
    
    df["canonical_family"] = mapping_results.apply(lambda x: x["canonical_family"])
    df["attack_techniques"] = mapping_results.apply(lambda x: x["techniques"])
    df["d3fend_controls"] = mapping_results.apply(lambda x: x["countermeasures"])
    
    # Calculate susceptibility score
    impact = (
        df[impact_column]
        if impact_column and impact_column in df.columns
        else policy.impact_default
    )
    
    df["susceptibility"] = df["probability"].clip(0.0, 1.0)
    df["susceptibility_bucket"] = pd.cut(
        df["susceptibility"],
        bins=[0.0, 0.33, 0.66, 1.0],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )
    df["expected_loss"] = df["susceptibility"] * float(impact)
    
    return df


def write_register(df: pd.DataFrame, name: str) -> None:
    settings = get_settings()
    settings.register_dir.mkdir(parents=True, exist_ok=True)
    settings.policies_dir.mkdir(parents=True, exist_ok=True)
    csv_path = settings.register_dir / f"{name}.csv"
    json_path = settings.register_dir / f"{name}.json"
    df.to_csv(csv_path, index=False)
    with open(json_path, "wb") as f:
        f.write(orjson.dumps(df.to_dict(orient="records")))
