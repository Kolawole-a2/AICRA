from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

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


def write_register(
    df: pd.DataFrame, 
    name: str, 
    model_id: str | None = None, 
    policy_id: str | None = None
) -> tuple[Path, Path]:
    """Write register to both latest and archived versions with metadata."""
    settings = get_settings()
    settings.register_dir.mkdir(parents=True, exist_ok=True)
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Add metadata columns if provided
    df_with_metadata = df.copy()
    if model_id:
        df_with_metadata["model_id"] = model_id
    if policy_id:
        df_with_metadata["policy_id"] = policy_id
    
    # Generate timestamp for archived version
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Write latest version to artifacts/
    latest_path = settings.artifacts_dir / "risk_register.csv"
    df_with_metadata.to_csv(latest_path, index=False)
    
    # Write archived version to artifacts/
    archived_path = settings.artifacts_dir / f"risk_register_{timestamp}.csv"
    df_with_metadata.to_csv(archived_path, index=False)
    
    # Also write to register directory for backward compatibility
    register_csv_path = settings.register_dir / f"{name}.csv"
    register_json_path = settings.register_dir / f"{name}.json"
    df_with_metadata.to_csv(register_csv_path, index=False)
    with open(register_json_path, "wb") as f:
        f.write(orjson.dumps(df_with_metadata.to_dict(orient="records")))
    
    return latest_path, archived_path
