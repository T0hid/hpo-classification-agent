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

Outputs are written to `results/` as a CSV table and a per-term
reasoning log.

## Repository layout

```
hpo-classification-agent/
  main.py                       Main pipeline
  requirements.txt              Python dependencies
  hpo_terms_to_classify.csv     List of HPO IDs to process
  README.md                     This file
  data/                         Created automatically; holds hp.obo
  results/                      Created automatically; holds outputs
```

## Setup and run

1. Clone or download this repository.

2. Install Python dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Set your OpenRouter API key as an environment variable. The script
   reads it from there so it never touches the source code:

   ```
   export OPENROUTER_API_KEY="your-key-here"
   ```

   On Windows PowerShell:

   ```
   $env:OPENROUTER_API_KEY="your-key-here"
   ```

4. Run the pipeline:

   ```
   python main.py
   ```

On the first run, the script automatically creates the `data/` and
`results/` folders, downloads the HPO ontology (`hp.obo`, about 10 MB),
and moves the input CSV into `data/`. No manual setup is required.

Outputs land in `results/`:

* `classifications.csv` — one row per HPO term with tier, category,
  severity, and management profile.
* `reasoning_log.json` — full trace of every ReAct step and the JSON
  classification returned by the LLM.

## Optional environment variables

| Name | Default | Purpose |
|------|---------|---------|
| `MODEL_NAME` | `openai/gpt-4o` | Any OpenRouter-compatible model id |
| `NCBI_EMAIL` | `reproducibility@example.org` | Identifier sent to NCBI E-utilities |

## Note on the classification prompt

The classification prompt embedded in `main.py` is a structural
template that preserves the input and output JSON schema of the
production system. The detailed clinical decision rules contained in
the production prompt have been licensed to UNSW Sydney and are not
included in this public release. The full production prompt is
available to editors and peer reviewers upon request during
manuscript assessment, and to academic researchers under a material
transfer agreement for non-commercial use, by contacting the
corresponding author.

## Reproducibility notes

* The ReAct planner and the final classifier both run with
  `temperature = 0.0` to suppress sampling randomness.
* PubMed retrieval depends on NCBI's live index, so the exact set of
  retrieved abstracts may drift over time. The reasoning log records
  every PubMed identifier used, which lets reviewers trace any
  individual classification back to its evidence.
