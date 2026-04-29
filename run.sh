#!/usr/bin/env bash
set -e

# Run the main pipeline
python -u main.py

# Run the visualization generator
python -u visualization.py

# Run the hyperparameter tuning
python -u hyperparameter.py

# Run the severity_classification
python -u severity_classification.py