# HPO Term Classification ReAct Agent

Reproducibility build accompanying the manuscript submission.

## What this code does

For each HPO (Human Phenotype Ontology) term in the input list, the agent
runs a ReAct loop that:

1. Retrieves rich HPO context (definition, synonyms, parents) from the
   local `hp.obo` ontology file.
2. Looks up the matching MeSH descriptor via NCBI E-utilities.
3. Generates targeted PubMed queries.
4. Searches PubMed and retrieves abstracts.
5. Sends the gathered evidence to an LLM that returns a structured
   classification (tier, category, severity, management profile) in
   strict JSON.

Downstream scripts then perform severity classification, hyperparameter
analysis, and figure generation.

## Repository layout

```
hpo-classification-agent/
  main.py                       Main classification pipeline
  severity_classification.py    Severity & management analysis
  hyperparameter.py             Hyperparameter sensitivity plots
  visualization.py              Figures 1-6 + validation dashboard
  requirements.txt              Python dependencies
  run.sh                        Convenience runner for all four scripts
  hpo_terms_to_classify.csv     Input list of HPO IDs
  scripts/
    download_public_data.py     Fetches hp.obo and phenotype.hpoa
  data/                         Inputs (see data/README.md)
  results/                      Created at runtime; outputs land here
```

## Setup

1. Clone the repository and install dependencies:

   ```bash
   git clone https://github.com/<your-username>/hpo-classification-agent.git
   cd hpo-classification-agent
   pip install -r requirements.txt
   ```

2. Download the public HPO data files:

   ```bash
   python scripts/download_public_data.py
   ```

3. Set your OpenRouter API key (never commit this):

   ```bash
   export OPENROUTER_API_KEY="your-key-here"
   ```

   On Windows PowerShell:

   ```powershell
   $env:OPENROUTER_API_KEY="your-key-here"
   ```

## Run

Run any of the four scripts independently, or all of them via `run.sh`:

```bash
python main.py                       # produces data/Classified_terms.csv (and results/)
python hyperparameter.py             # uses data/hyperparameter_results.csv
python visualization.py              # needs Classified_terms.csv + GT_file.xlsx
python severity_classification.py    # needs the IP-protected files (see below)
```

Each script auto-detects whether it is running on Code Ocean (which uses
fixed `/data` and `/results` mount points) or in a regular checkout, and
chooses paths accordingly. No code edits are needed.

Optional environment variables:

| Name | Default | Purpose |
|---|---|---|
| `MODEL_NAME` | `openai/gpt-4o` | Any OpenRouter-compatible model id |
| `NCBI_EMAIL` | `reproducibility@example.org` | Identifier sent to NCBI E-utilities |

## Data availability

This project distinguishes three categories of data:

**1. Public data — fetched automatically.**
`hp.obo` and `phenotype.hpoa` are released by the HPO consortium under
CC-BY-4.0 and are downloaded by `scripts/download_public_data.py`. They
are not committed to the repository.

**2. Small project files — committed to the repository.**
`hpo_terms_to_classify.csv`, `hyperparameter_results.csv`, `GT_file.xlsx`,
`MM_with_Severity.csv`, `CS.xlsx`, `SF.xlsx`. All are well under 1 MB and
ship with the repo.

**3. IP-protected derived datasets — not redistributed.**
The following files are products of the production pipeline and contain
proprietary classification output. They are not included in this
repository or in any public archive:

* `Classified_terms.csv` (~69 MB)
* `merged_gene_disease_phenotype.csv` (~49 MB)
* `final_output_with_severity.csv` (~59 MB)

These are available from the corresponding author upon reasonable
request, subject to a Material Transfer / Data Use Agreement with UNSW
Sydney for non-commercial academic use.

`Classified_terms.csv` can also be regenerated from scratch by anyone
running `main.py` with their own OpenRouter API key — the file is
produced deterministically (temperature = 0).

## Note on the classification prompt

The classification prompt embedded in `main.py` is a structural template
that preserves the input and output JSON schema of the production system.
The detailed clinical decision rules contained in the production prompt
have been licensed to UNSW Sydney and are not included in this public
release. The full production prompt is available to editors and peer
reviewers upon request during manuscript assessment, and to academic
researchers under a material transfer agreement for non-commercial use,
by contacting the corresponding author.

## Reproducibility notes

* Both the ReAct planner and the final classifier run with
  `temperature = 0.0` to suppress sampling randomness.
* PubMed retrieval depends on NCBI's live index, so the exact set of
  retrieved abstracts may drift over time. The reasoning log
  (`results/reasoning_log.json`) records every PubMed identifier used,
  which lets reviewers trace any classification back to its evidence.
* The exact model snapshot used in the manuscript is logged in each row
  of `Classified_terms.csv`. To reproduce the published numbers, set
  `MODEL_NAME` to that snapshot.

## Citation

If you use this code, please cite the accompanying manuscript (citation
will be added upon publication) and archive the version of this
repository you ran via the Zenodo–GitHub integration (recommended for
formal citation with a DOI).

## License

Code: MIT (see `LICENSE`).
HPO data files: CC-BY-4.0 (HPO consortium).
IP-protected datasets: not licensed for redistribution.
