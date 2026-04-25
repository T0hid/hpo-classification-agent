# HPO Term Classification ReAct Agent

Reproducibility build accompanying the manuscript submission.

## What this code does

For each HPO (Human Phenotype Ontology) term in the input list, the
agent runs a ReAct loop that:

1. Retrieves rich HPO context (definition, synonyms, parents) from the
   local `hp.obo` ontology file.
2. Looks up the matching MeSH descriptor via NCBI E-utilities.
3. Generates targeted PubMed queries.
4. Searches PubMed and retrieves abstracts.
5. Sends the gathered evidence to an LLM that returns a structured
   classification (tier, category, severity, management profile) in
   strict JSON.

Outputs land in `/results` as a CSV table and a per-term reasoning log.

## Files

| Path | Purpose |
|------|---------|
| `code/main.py` | Main pipeline (single-threaded, reviewer-friendly) |
| `code/run` | CodeOcean entry point |
| `data/hp.obo` | HPO ontology (downloaded from hpo.jax.org) |
| `data/hpo_terms_to_classify.csv` | List of HPO IDs to process |
| `environment/postInstall` | Installs Python dependencies |
| `requirements.txt` | Reference dependency list |

## Required CodeOcean Secret

The pipeline calls a hosted LLM. The API key is read from an
environment variable so it never touches the source code or the git
history.

In the CodeOcean capsule, open **Environment > Secrets** and add:

| Name | Value |
|------|-------|
| `OPENROUTER_API_KEY` | your OpenRouter key |

Optional environment variables:

| Name | Default | Notes |
|------|---------|-------|
| `MODEL_NAME` | `openai/gpt-4o` | Any OpenRouter-compatible model id |
| `NCBI_EMAIL` | `reproducibility@example.org` | Used to identify the caller to NCBI |

## How to run on CodeOcean

1. Add the `OPENROUTER_API_KEY` secret (see above).
2. Click **Reproducible Run**.

CodeOcean will execute `code/run`, which calls `main.py`. Outputs are
written to `/results/`:

* `classifications.csv` – one row per HPO term with tier, category,
  severity, and management profile.
* `reasoning_log.json` – full trace of every ReAct step and the JSON
  classification returned by the LLM.

## How to run locally

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY="your-key"
python code/main.py
```

Outputs are written to `../results/` relative to `code/main.py`.

## Note on the classification prompt

The classification prompt in `main.py` is a structural template that
preserves the production system's input and output JSON schema. The
full production prompt (containing the detailed clinical exclusion
filters, decision trees, and citation enforcement rules) is available
from the corresponding author on reasonable request, subject to a
data-use agreement.

## Reproducibility notes

* All randomness is suppressed: temperature is set to `0.0` for both
  the ReAct planner and the final classifier.
* PubMed retrieval depends on NCBI's live index, so the exact set of
  retrieved abstracts may drift over time. The reasoning log records
  every PubMed identifier used, which lets reviewers trace any
  individual classification back to its evidence.
