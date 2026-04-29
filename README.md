# HPO Term Classification ReAct Agent

Reproducibility build accompanying the manuscript submission.

## Overview

This pipeline classifies Human Phenotype Ontology (HPO) terms into clinical
severity tiers and management profiles using a **ReAct (Reasoning + Acting)
agent** backed by a large language model. The classification framework follows
tier and category definitions consistent with **ACOG- and ACMG-endorsed
guidelines** for reproductive carrier screening and genomic medicine.

For every input HPO term, the agent produces:

- A **classification tier** (1–4 / NFC) and category
- A **Quality of Life assessment** (Affected / Not Affected)
- A **management profile** (surgical, pharmacological, palliative, monitoring,
  dietary, assistive, therapeutic support, or none)
- A **structured reasoning trace** with `[Ref N]` citations to PubMed evidence
- A **source-verification score** — every cited claim is fact-checked by an
  LLM judge against its source abstract and labelled `direct`,
  `valid_inference`, `weak_inference`, or `unsupported`

Downstream scripts then perform aggregate severity analysis, hyperparameter
sensitivity testing, and figure / dashboard generation for manuscript
reproducibility.

## What each script does

| Script | Purpose |
|---|---|
| **`main.py`** | Main classification ReAct agent. Wraps the LLM in a templated prompt that (1) retrieves rich HPO context — definition, synonyms, parents — from the local `hp.obo`, (2) maps the term to a MeSH descriptor via NCBI E-utilities, (3) generates targeted PubMed queries, (4) retrieves abstracts, (5) classifies the term per ACOG/ACMG-endorsed tier rules, and (6) runs an LLM-based **source-verification** loop that scores how well each `[Ref N]` claim is supported by its cited abstract. Runs on the supplied `hpo_terms_to_classify.csv`, or any HPO ID list you provide. |
| **`visualization.py`** | Generates Figures 1–6 of the manuscript and the validation dashboard from the output of `main.py`. Computes accuracy, F1, MCC, kappa, PSI, and confusion matrices against the held-out ground-truth file. |
| **`hyperparameter.py`** | Hyperparameter sensitivity analysis. Renders the model × temperature accuracy / response-time heatmap. |
| **`severity_classification.py`** | Computes the final per–gene-disease severity classification (Profound / Severe / Moderate / Mild) by aggregating tiers across phenotypes, applies a 30 % frequency filter, breaks results down by body system and mode of inheritance, and **compares against the ACMG carrier-screening (CS) list, the ACMG Secondary Findings (SF) list, and the Mackenzie's Mission (MM) reproductive-carrier panel**. |

## Repository layout

```
hpo-classification-agent/
  main.py                        Main classification pipeline (ReAct agent)
  severity_classification.py     Aggregate severity + MM / SF / CS comparison
  hyperparameter.py              Hyperparameter sensitivity heatmap
  visualization.py               Figures 1-6 + validation dashboard
  requirements.txt               Python dependencies
  run.sh                         Convenience runner for all four scripts
  hpo_terms_to_classify.csv      Default input list of HPO IDs
  scripts/
    download_public_data.py      Fetches hp.obo from the HPO consortium
  data/                          Input data (see data/README.md)
  results/                       Created at runtime; outputs land here
```

## Setup

1. Clone the repository and install dependencies:

   ```bash
   git clone https://github.com/<your-username>/hpo-classification-agent.git
   cd hpo-classification-agent
   pip install -r requirements.txt
   ```

2. Download `hp.obo` (the only public data file needed):

   ```bash
   python scripts/download_public_data.py
   ```

   `main.py` will also fetch `hp.obo` automatically on its first run if it
   isn't present.

3. Set your OpenRouter API key (never commit this — `main.py` is the only
   script that needs it):

   ```bash
   export OPENROUTER_API_KEY="your-key-here"
   ```

   On Windows PowerShell:

   ```powershell
   $env:OPENROUTER_API_KEY="your-key-here"
   ```

## Running the pipeline

Each script can be run independently, or all of them in sequence via `run.sh`:

```bash
python main.py                       # produces the classification output + reasoning log
python hyperparameter.py             # hyperparameter sensitivity heatmap
python severity_classification.py    # aggregate severity + MM / SF / CS comparison
python visualization.py              # Figures 1-6 + validation dashboard
```

`hyperparameter.py` and `severity_classification.py` run on the data already
included in `data/` — no external downloads, no API key. `visualization.py`
needs the output of `main.py` plus the ground-truth file `GT_file.xlsx`.

### Classifying your own HPO terms

`main.py` reads HPO IDs from `data/hpo_terms_to_classify.csv` (a single column
named `hpo_id`, optionally with an `hpo_name` column). Replace that file with
your own list and re-run `main.py` to classify any set of HPO terms.

### Optional environment variables

| Name | Default | Purpose |
|---|---|---|
| `MODEL_NAME` | `openai/gpt-4o` | Any OpenRouter-compatible model id |
| `NCBI_EMAIL` | `reproducibility@example.org` | Identifier sent to NCBI E-utilities |

## Data availability

- **Public data.** `hp.obo` (HPO consortium, CC-BY-4.0) is downloaded on
  demand by `scripts/download_public_data.py` or by `main.py` on first run.
- **Project data.** All small-to-medium files needed to run
  `hyperparameter.py`, `severity_classification.py`, and the validation step
  of `visualization.py` are committed to `data/`. See `data/README.md` for
  the full inventory.
- **IP-protected output.** The full classification output of `main.py`
  contains the per-term reasoning text, `[Ref N]` citations, and verification
  breakdown. Because this output embeds proprietary classification logic, it
  is **not redistributed in this repository** and is available from the
  corresponding author on reasonable request — including to peer reviewers
  and editors during manuscript assessment, and to academic researchers under
  a Material Transfer / Data Use Agreement with UNSW Sydney for non-commercial
  use. It can also be **regenerated from scratch** by anyone running
  `main.py` with their own OpenRouter API key — the agent runs
  deterministically (`temperature = 0`).

## Note on the classification prompt

The classification prompt embedded in `main.py` is a **structural template**
that preserves the exact input and output JSON schema of the production
system. The detailed clinical decision rules (exclusion filters, decision
trees, citation-enforcement language) contained in the production prompt have
been licensed to UNSW Sydney and are not included in this public release. The
full production prompt is available to editors and peer reviewers on request
during manuscript assessment, and to academic researchers under a Material
Transfer Agreement for non-commercial use, by contacting the corresponding
author.

## Reproducibility notes

- Both the ReAct planner and the final classifier run with `temperature = 0.0`
  to suppress sampling randomness.
- PubMed retrieval depends on NCBI's live index, so the exact set of retrieved
  abstracts may drift over time. The reasoning log records every PubMed
  identifier the agent used, which lets reviewers trace any classification
  back to its evidence.
- Source verification is itself an LLM-based step (a "judge" prompt that
  classifies each `[Ref N]` claim as `direct`, `valid_inference`,
  `weak_inference`, or `unsupported`). The full per-claim verification log
  is stored in the reasoning log.
- The exact model snapshot used for the manuscript figures is logged in the
  classification output. To reproduce the published numbers, set
  `MODEL_NAME` to that snapshot.

## Citation

If you use this code, please cite the accompanying manuscript (citation will
be added upon publication) and archive the version of this repository you ran
via the Zenodo–GitHub integration (recommended for formal citation with a
DOI).

## License

- **Code:** MIT (see `LICENSE`).
- **HPO data files:** CC-BY-4.0 (HPO consortium).
- **IP-protected datasets:** not licensed for redistribution.
