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
- A **severity assessment** (Severe / Non-Severe)
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

| Script | Purpose | Inputs (in `data/`) | Outputs |
|---|---|---|---|
| **`main.py`** | Main classification ReAct agent. Wraps the LLM in a templated prompt that (1) retrieves rich HPO context — definition, synonyms, parents — from the local `hp.obo`, (2) maps the term to a MeSH descriptor via NCBI E-utilities, (3) generates targeted PubMed queries, (4) retrieves abstracts, (5) classifies the term per ACOG/ACMG-endorsed tier rules, and (6) runs an LLM-based **source-verification** loop that scores how well each `[Ref N]` claim is supported by its cited abstract. Runs on the supplied `hpo_terms_to_classify.csv`, or any HPO ID list you provide. | `hp.obo`, `hpo_terms_to_classify.csv` | `Classified_terms.csv`, `reasoning_log.json` |
| **`visualization.py`** | Generates Figures 1–6 of the manuscript and the validation dashboard from the classification output. Computes accuracy, F1, MCC, kappa, PSI, and confusion matrices against the held-out ground-truth file. The full classification output it consumes (with reasoning text and citation breakdown) is **not redistributed publicly** and is available on reasonable request from the corresponding author. | `Classified_terms.csv` (regenerable from `main.py`), `GT_file.xlsx` | Figures 1–6 PDFs, `comprehensive_dashboard.pdf`, `comprehensive_validation_report.txt`, analysis report |
| **`hyperparameter.py`** | Hyperparameter sensitivity analysis. Renders the model × temperature accuracy / response-time heatmap. | `hyperparameter_results.csv` | `3_combined_performance_heatmap.pdf` |
| **`severity_classification.py`** | Computes the final per–gene-disease severity classification (Profound / Severe / Moderate / Mild) by aggregating tiers across phenotypes, applies a 30 % frequency filter, breaks results down by body system and mode of inheritance, and **compares against the ACMG carrier-screening (CS) list, the ACMG Secondary Findings (SF) list, and the Mackenzie's Mission (MM) reproductive-carrier panel**. | `final_merged_output_with_freq.pkl`, `MM_with_Severity.csv`, `CS.xlsx`, `SF.xlsx` | `gene_severity_classification_excluding_nfc.csv`, `combined_analysis_panel.pdf`, `comprehensive_report.txt`, `severity_distribution_summary.txt` |

> **Note on `phenotype.hpoa`** — earlier versions of the pipeline depended on
> the `phenotype.hpoa` annotation file. **It is no longer required by any of
> the four scripts.** You can ignore it even if `download_public_data.py`
> still fetches it for backward compatibility.

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
    final_merged_output_with_freq.pkl  Merged gene-disease-phenotype file (~24 MB, shipped)
    MM_with_Severity.csv         Mackenzie's Mission gene list with severity (shipped)
    CS.xlsx                      ACMG carrier-screening list (shipped)
    SF.xlsx                      ACMG Secondary Findings list (shipped)
    GT_file.xlsx                 Ground-truth labels for validation (shipped)
    hyperparameter_results.csv   Pre-computed hyperparameter sweep (shipped)
    hpo_terms_to_classify.csv    Default input list (also at repo root)
  results/                       Created at runtime; outputs land here
```

## Setup

1. Clone the repository and install dependencies:

   ```bash
   git clone https://github.com/<your-username>/hpo-classification-agent.git
   cd hpo-classification-agent
   pip install -r requirements.txt
   ```

2. Download `hp.obo` (the only required public data file):

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

You can run each script independently, or all of them in sequence via `run.sh`:

```bash
python main.py                       # produces Classified_terms.csv + reasoning log
python hyperparameter.py             # uses bundled hyperparameter_results.csv
python severity_classification.py    # uses bundled final_merged_output_with_freq.pkl
python visualization.py              # needs Classified_terms.csv + GT_file.xlsx
```

`hyperparameter.py` and `severity_classification.py` run **out of the box** on
the data shipped with the repository — no external downloads, no API key.
`visualization.py` needs `Classified_terms.csv`, which you generate by running
`main.py` first (or by requesting it from the corresponding author — see
*Data availability* below).

### Classifying your own HPO terms

`main.py` reads HPO IDs from `data/hpo_terms_to_classify.csv` (a single column
named `hpo_id`, optionally with an `hpo_name` column). Replace that file with
your own list and re-run `main.py` to classify any set of HPO terms.

Each script auto-detects whether it is running in a Code Ocean capsule (which
uses fixed `/data` and `/results` mount points) or in a regular checkout, and
chooses paths accordingly. No code edits are needed.

### Optional environment variables

| Name | Default | Purpose |
|---|---|---|
| `MODEL_NAME` | `openai/gpt-4o` | Any OpenRouter-compatible model id |
| `NCBI_EMAIL` | `reproducibility@example.org` | Identifier sent to NCBI E-utilities |

## Data availability

This project distinguishes three categories of data:

**1. Public data — fetched on demand.**
`hp.obo` is released by the HPO consortium under CC-BY-4.0 and is downloaded
automatically by `scripts/download_public_data.py` (or by `main.py` on first
run). It is not committed to the repository. **`phenotype.hpoa` is no longer
required by any of the four scripts** even though `download_public_data.py`
still fetches it.

**2. Project data — shipped in the repository.**
The following files are committed under `data/` and are sufficient to run
`hyperparameter.py`, `severity_classification.py`, and the validation parts of
`visualization.py`:

- `hpo_terms_to_classify.csv` — default HPO ID input list
- `hyperparameter_results.csv` — model × temperature sweep results (~5 KB)
- `final_merged_output_with_freq.pkl` — merged gene-disease-phenotype file
  with frequency annotations (~24 MB)
- `MM_with_Severity.csv` — Mackenzie's Mission carrier panel with severity
  (~219 KB)
- `CS.xlsx` — ACMG carrier-screening list (~14 KB)
- `SF.xlsx` — ACMG Secondary Findings list (~17 KB)
- `GT_file.xlsx` — ground-truth labels for validation (~60 KB)

**3. IP-protected derived dataset — not redistributed.**
The classification output produced by `main.py`,

- `Classified_terms.csv` (~69 MB),

contains the **full per-term reasoning text, `[Ref N]` citations, and
verification breakdown**. Because this output embeds proprietary classification
logic, **it is not included in the public repository or in any public archive**.
It is available from the corresponding author on reasonable request — including
to peer reviewers and editors during manuscript assessment, and to academic
researchers under a Material Transfer / Data Use Agreement with UNSW Sydney for
non-commercial use.

`Classified_terms.csv` can also be **regenerated from scratch** by anyone
running `main.py` with their own OpenRouter API key — the agent runs
deterministically (`temperature = 0`).

## Note on the classification prompt

The classification prompt embedded in `main.py` is a **structural template**
that preserves the exact input and output JSON schema of the production system.
The detailed clinical decision rules (exclusion filters, decision trees,
citation-enforcement language) contained in the production prompt have been
licensed to UNSW Sydney and are not included in this public release. The full
production prompt is available to editors and peer reviewers on request during
manuscript assessment, and to academic researchers under a Material Transfer
Agreement for non-commercial use, by contacting the corresponding author.

## Reproducibility notes

- Both the ReAct planner and the final classifier run with `temperature = 0.0`
  to suppress sampling randomness.
- PubMed retrieval depends on NCBI's live index, so the exact set of retrieved
  abstracts may drift over time. `reasoning_log.json` records every PubMed
  identifier the agent used, which lets reviewers trace any classification
  back to its evidence.
- Source verification is itself an LLM-based step (a "judge" prompt that
  classifies each `[Ref N]` claim as `direct`, `valid_inference`,
  `weak_inference`, or `unsupported`). The full per-claim verification log is
  stored in `reasoning_log.json`.
- The exact model snapshot used for the manuscript figures is logged in each
  row of `Classified_terms.csv`. To reproduce the published numbers, set
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
