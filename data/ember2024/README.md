Use Thrember to download the full EMBER-2024 dataset here.

Install Thrember:
pip install "git+https://github.com/FutureComputing4AI/EMBER2024.git"

Download:
python - <<END
import os
import thrember
os.makedirs("data/ember2024", exist_ok=True)
thrember.download_dataset("data/ember2024")
END

Expected files:
- train/ember_features.jsonl
- test/ember_features.jsonl

