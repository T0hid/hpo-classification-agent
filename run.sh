#!/usr/bin/env bash
# Convenience runner. Each script also runs fine on its own.
set -e

cd "$(dirname "$0")"

# Make sure public data is available (idempotent)
python scripts/download_public_data.py

# Run the main classification pipeline
python -u main.py

# Run hyperparameter analysis
python -u hyperparameter.py

# Run severity classification (needs IP-protected files; see README)
python -u severity_classification.py

# Run figure / validation pipeline
python -u visualization.py
