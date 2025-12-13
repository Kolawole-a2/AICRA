# Data Availability (EMBER 2024)

This repository does **not** include the full EMBER-2024 JSONL dataset in Git history or in the repository contents.

## Why the dataset is not in Git
- The EMBER JSONL splits are very large (GB-scale) and will break typical GitHub pushes/timeouts.
- Keeping large datasets out of Git also reduces accidental exposure and improves reproducibility hygiene.

## Expected local location
Place the EMBER-2024 JSONL files locally at:

- `data/ember2024_real/`

Example pattern:
- `*_train.jsonl`

## Configure via environment variable (recommended)
You can override the dataset location using:

- `AICRA_EMBER2024_DIR`

Example:
- Windows PowerShell: `$env:AICRA_EMBER2024_DIR="D:\data\ember2024_real"`
- Bash: `export AICRA_EMBER2024_DIR="/mnt/d/data/ember2024_real"`

## How experiments depend on this data
- **H1** (Static PE features → ransomware susceptibility): requires EMBER JSONL splits to build feature matrices and labels.
- **H2** (Calibration & transferability): requires validation splits for isotonic calibration evaluation.
- **H3** (Mapping comparison): does not require full EMBER JSONL, but may consume outputs generated from H1/H2.

## Safety: do not re-add JSONL to Git
The repository `.gitignore` intentionally excludes:
- `data/`, and especially `data/ember2024_real/`
- `*.jsonl`

If you see JSONL files listed in `git status`, stop and verify `.gitignore`.
