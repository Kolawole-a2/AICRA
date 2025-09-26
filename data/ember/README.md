Place real EMBER-2024 files here after download.

Expected paths:
- train/ember_features.jsonl
- test/ember_features.jsonl

Use:
python src/download_ember.py --outdir data/ember --datasets train test --limit 50000 --force --base-url <URL>
python src/feature_extractor.py --input data/ember/train/ember_features.jsonl --out data/ember/train_pe.npz --limit 50000 --force

