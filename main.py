"""
HPO Term Classification ReAct Agent (Reproducibility Build)
============================================================

Companion code for the manuscript submission.

This is a minimal, single-threaded build intended for reviewers and
CodeOcean reproducibility checks. It removes the production caching
and checkpoint layers and replaces the proprietary classification
prompt with a structural template (see PROMPT_TEMPLATE_NOTE below).

Required environment variable:
    OPENROUTER_API_KEY   API key for the LLM endpoint (set as a
                         CodeOcean Secret, never committed to code).

Inputs (in /data):
    hp.obo                          HPO ontology
    hpo_terms_to_classify.csv       List of HPO IDs to process

Outputs (in /results):
    classifications.csv             Tabular classification results
    reasoning_log.json              Full per-term reasoning trace
"""

import os
import sys
import json
import time
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pronto
import requests
from Bio import Entrez

warnings.filterwarnings("ignore")


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

# Detect whether we are running inside a CodeOcean capsule (paths are fixed
# there) or on a local machine (paths are relative to this file).
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# If the input CSV is at the repo root (where it ships from GitHub),
# move it into data/ on first run.
_root_csv = os.path.join(HERE, "hpo_terms_to_classify.csv")
_data_csv = os.path.join(DATA_DIR, "hpo_terms_to_classify.csv")
if os.path.exists(_root_csv) and not os.path.exists(_data_csv):
    import shutil
    shutil.move(_root_csv, _data_csv)
    print(f"Moved input CSV into {DATA_DIR}/")

# If the HPO ontology is missing, download it automatically.
_obo_path = os.path.join(DATA_DIR, "hp.obo")
if not os.path.exists(_obo_path):
    import urllib.request
    print("Downloading hp.obo from purl.obolibrary.org (about 10 MB)...")
    urllib.request.urlretrieve("http://purl.obolibrary.org/obo/hp.obo", _obo_path)
    print(f"Saved to {_obo_path}")

INPUT_CSV_PATH = os.path.join(DATA_DIR, "hpo_terms_to_classify.csv")
HPO_ONTOLOGY_PATH = os.path.join(DATA_DIR, "hp.obo")

OUTPUT_CSV_PATH = os.path.join(RESULTS_DIR, "classifications.csv")
REASONING_LOG_PATH = os.path.join(RESULTS_DIR, "reasoning_log.json")

# API key is read from environment only. Never hardcode it.
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = os.environ.get("MODEL_NAME", "openai/gpt-4o")

# Required for NCBI / PubMed access. Override via env if you wish.
Entrez.email = os.environ.get("NCBI_EMAIL", "reproducibility@example.org")

MAX_REACT_ITERATIONS = 10
REACT_TEMPERATURE = 0.0
PUBMED_MAX_RESULTS = 10


# =============================================================================
# 2. ReAct AGENT DATA STRUCTURES
# =============================================================================

class AgentAction(Enum):
    GET_HPO_CONTEXT = "get_hpo_context"
    GET_MESH_TERM = "get_mesh_term"
    TRANSFORM_QUERY = "transform_query"
    SEARCH_PUBMED = "search_pubmed"
    CLASSIFY = "classify"
    INSUFFICIENT_INFO = "insufficient_info"


@dataclass
class ReActStep:
    iteration: int
    thought: str
    action: AgentAction
    observation: str


@dataclass
class ReActContext:
    hpo_id: str
    hpo_context: Optional[Dict] = None
    mesh_term: Optional[str] = None
    mesh_term_searched: bool = False
    transformed_queries: List[str] = field(default_factory=list)
    queries_transformed: bool = False
    pubmed_articles: List[Dict] = field(default_factory=list)
    pubmed_searched: bool = False
    steps: List[ReActStep] = field(default_factory=list)
    final_classification: Optional[Dict] = None
    reference_urls: Dict[str, str] = field(default_factory=dict)


# =============================================================================
# 3. LLM CALL
# =============================================================================

def call_llm(session: requests.Session, user_prompt: str, system_prompt: str,
             temperature: float = 0.0) -> Optional[Dict]:
    """Single LLM call returning parsed JSON, or None on failure."""
    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY environment variable is not set.")
        return None

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://codeocean.com",
        "X-Title": "HPO Classification Reproducibility Build",
    }
    payload = {
        "model": MODEL_NAME,
        "response_format": {"type": "json_object"},
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    for attempt in range(4):
        try:
            response = session.post(OPENROUTER_API_URL, json=payload,
                                    headers=headers, timeout=180)
            if response.status_code == 429:
                wait = 5 * (2 ** attempt)
                print(f"  Rate limited. Sleeping {wait}s.")
                time.sleep(wait)
                continue
            response.raise_for_status()
            return json.loads(response.json()["choices"][0]["message"]["content"])
        except Exception as e:
            print(f"  LLM call failed (attempt {attempt + 1}): {e}")
            time.sleep(3)
    return None


# =============================================================================
# 4. TOOL FUNCTIONS
# =============================================================================

def load_hpo_ontology(obo_path: str) -> Optional[pronto.Ontology]:
    print(f"Loading HPO ontology from {obo_path}")
    if not os.path.exists(obo_path):
        print(f"ERROR: Ontology file not found at {obo_path}")
        return None
    return pronto.Ontology(obo_path)


def get_hpo_context(hpo_id: str, ontology: pronto.Ontology) -> Dict[str, Any]:
    try:
        term = ontology[hpo_id]
        return {
            "id": hpo_id,
            "name": term.name,
            "definition": term.definition or "",
            "synonyms": [s.description for s in term.synonyms if s.scope == "EXACT"],
            "comment": term.comment or "",
            "parents": [p.name for p in term.superclasses(distance=1, with_self=False)],
        }
    except KeyError:
        return {"error": f"Term {hpo_id} not found in ontology."}


def get_mesh_term(hpo_name: str, synonyms: List[str]) -> Optional[str]:
    """Best-effort MeSH lookup. Returns None if nothing usable is found."""
    for term in [hpo_name] + (synonyms or []):
        try:
            handle = Entrez.esearch(db="mesh", term=term.strip(), retmax=1)
            result = Entrez.read(handle)
            handle.close()
            if result.get("IdList"):
                mesh_id = result["IdList"][0]
                try:
                    fetch = Entrez.efetch(db="mesh", id=mesh_id, retmode="xml")
                    xml_data = fetch.read()
                    fetch.close()
                    if xml_data and not xml_data.startswith(b"<!DOCTYPE"):
                        root = ET.fromstring(xml_data)
                        node = root.find(".//DescriptorName/String")
                        if node is not None and node.text:
                            return node.text
                    return f"MeSH_ID:{mesh_id}"
                except Exception:
                    return f"MeSH_ID:{mesh_id}"
        except Exception:
            continue
    return None


def transform_query(session: requests.Session, hpo_context: Dict,
                    mesh_term: Optional[str]) -> List[str]:
    """Generate a small set of targeted PubMed queries via the LLM."""
    name = hpo_context.get("name", "")
    brainstorm_prompt = (
        f'The HPO term is "{name}". List up to 3 common clinical synonyms or '
        f'closely related diagnostic terms. Respond as JSON: '
        f'{{"related_terms": ["term1", "term2"]}}.'
    )
    response = call_llm(session, brainstorm_prompt,
                        "You are a medical terminologist.", temperature=0.0)
    related = response.get("related_terms", []) if response else []

    base_terms = list(dict.fromkeys([name] + related + hpo_context.get("synonyms", [])))
    if mesh_term and "MeSH_ID" not in str(mesh_term):
        base_terms.append(mesh_term)

    queries = []
    for term in base_terms:
        queries.extend([
            f'"{term}" congenital OR genetic syndrome',
            f'"{term}" AND (hereditary OR familial)',
            f'"{term}" pediatric OR neonatal onset',
        ])
    return list(dict.fromkeys(queries))[:12]


def search_pubmed(hpo_name: str, synonyms: List[str],
                  mesh_term: Optional[str]) -> List[Dict[str, str]]:
    """Cascading PubMed search with abstract retrieval."""
    print(f"  Searching PubMed for {hpo_name}")
    inherited_filter = (
        "(congenital[Title/Abstract] OR inherited[Title/Abstract] OR "
        "hereditary[Title/Abstract] OR genetic[Title/Abstract] OR "
        "familial[Title/Abstract] OR syndrome[Title/Abstract])"
    )
    all_terms = list(set([hpo_name] + (synonyms or [])))
    term_query = " OR ".join([f'"{t}"[Title/Abstract]' for t in all_terms])

    strategies = []
    if mesh_term and "MeSH_ID" not in str(mesh_term):
        strategies.append(f'("{mesh_term}"[MeSH Terms]) AND {inherited_filter}')
    strategies.append(f"({term_query}) AND {inherited_filter}")
    if mesh_term and "MeSH_ID" not in str(mesh_term):
        strategies.append(f'"{mesh_term}"[MeSH Terms]')
    strategies.append(f"({term_query})")

    article_ids: List[str] = []
    for query in strategies:
        if article_ids:
            break
        try:
            handle = Entrez.esearch(db="pubmed", term=query,
                                    retmax=str(PUBMED_MAX_RESULTS * 2),
                                    sort="relevance")
            record = Entrez.read(handle)
            handle.close()
            time.sleep(0.4)
            article_ids = record.get("IdList", [])
        except Exception as e:
            print(f"  PubMed search error: {e}")
            continue

    if not article_ids:
        return []

    try:
        fetch = Entrez.efetch(db="pubmed", id=article_ids,
                              rettype="abstract", retmode="xml")
        xml_data = fetch.read()
        fetch.close()
        time.sleep(0.4)
        root = ET.fromstring(xml_data)
    except Exception as e:
        print(f"  PubMed fetch error: {e}")
        return []

    articles = []
    for article_el in root.findall("PubmedArticle"):
        medline = article_el.find("MedlineCitation")
        if medline is None:
            continue
        pmid_el = medline.find("PMID")
        pmid = pmid_el.text if pmid_el is not None else "N/A"
        info = medline.find("Article")
        if info is None:
            continue
        title_el = info.find("ArticleTitle")
        title = title_el.text if title_el is not None else "No Title"
        abstract = " ".join([el.text for el in info.findall(".//AbstractText")
                             if el.text])
        if abstract and len(abstract) > 100:
            articles.append({
                "source": "PubMed",
                "id": pmid,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "title": title,
                "abstract": abstract,
            })
    return articles[:PUBMED_MAX_RESULTS]


# =============================================================================
# 5. PROMPTS
# =============================================================================
#
# PROMPT_TEMPLATE_NOTE
# --------------------
# The classification prompt below is a STRUCTURAL TEMPLATE released with
# this reproducibility build. It preserves the exact input and output
# JSON schemas of the production prompt so downstream parsing is
# unchanged. The full production prompt (containing the detailed
# clinical exclusion filters, decision trees, and citation enforcement
# rules) is available from the corresponding author on reasonable
# request.
# =============================================================================

def create_planning_prompt(context: ReActContext) -> str:
    state = f"HPO Term: {context.hpo_id}\n"
    state += f"  HPO context retrieved: {bool(context.hpo_context)}\n"
    state += f"  MeSH term searched: {context.mesh_term_searched}"
    state += f" (found: {context.mesh_term})\n" if context.mesh_term_searched else "\n"
    state += f"  Queries transformed: {context.queries_transformed}\n"
    state += f"  PubMed searched: {context.pubmed_searched}"
    state += f" ({len(context.pubmed_articles)} results)\n" if context.pubmed_searched else "\n"

    return f"""You are a ReAct agent classifying an HPO term.

Current state:
{state}

Available actions:
  get_hpo_context, get_mesh_term, transform_query, search_pubmed,
  classify, insufficient_info

Decision rules (follow in order):
  1. If HPO context not retrieved, use get_hpo_context.
  2. If MeSH not searched, use get_mesh_term.
  3. If queries not transformed, use transform_query.
  4. If PubMed not searched, use search_pubmed.
  5. Otherwise, use classify (or insufficient_info if 0 sources).

Respond as JSON:
{{"thought": "...", "action": "<action_name>"}}
"""


def create_classification_prompt(context: ReActContext) -> str:
    """
    Returns the classification prompt as a JSON string.

    NOTE: This is the structural template version. See PROMPT_TEMPLATE_NOTE
    above. The schema of the output is identical to the production version.
    """
    rich = context.hpo_context or {}
    sources = []
    refs = {}
    for i, art in enumerate(context.pubmed_articles):
        ref_id = f"[Ref {i + 1}]"
        sources.append({
            "reference_id": ref_id,
            "source_type": art.get("source", "Unknown"),
            "article_title": art.get("title", "N/A"),
            "url": art.get("url", ""),
            "content_snippet": art.get("abstract", "N/A"),
        })
        refs[ref_id] = f"{art.get('source', 'Unknown')}: {art.get('url', '')}"
    context.reference_urls = refs

    prompt = {
        "role": (
            "You are a clinical expert classifying a Human Phenotype Ontology "
            "(HPO) term using the provided literature snippets."
        ),
        "task": (
            "Analyse the HPO term in the context of the provided sources, then "
            "return a single JSON object that follows required_output_format."
        ),
        "input_data": {
            "hpo_term_to_classify": {
                "id": rich.get("id", "N/A"),
                "name": rich.get("name", "N/A"),
                "definition": rich.get("definition", "N/A"),
                "synonyms": rich.get("synonyms", []),
                "comment": rich.get("comment", "N/A"),
            },
            "retrieved_clinical_sources": sources or [
                {"reference_id": "N/A",
                 "content_snippet": "No clinical sources retrieved."}
            ],
        },
        "general_instructions": [
            "Base every factual statement strictly on the provided sources.",
            "Attach a [Ref N] tag to every claim that is derived from a source.",
            "If information is not present in the sources, write "
            "'Information not available in provided sources' instead of guessing.",
            "Assign exactly one classification category and one severity tier.",
        ],
        "allowed_categories": [
            "Shortened life span: infancy",
            "Shortened life span: childhood/adolescence",
            "Shortened life span: adulthood",
            "Intellectual disability",
            "Internal physical malformations",
            "Impaired mobility",
            "Dysmorphic feature",
            "Sensory Impairment - Vision",
            "Sensory Impairment - Hearing",
            "Sensory impairment: touch, others",
            "Immunodeficiency/cancer",
            "Mental illness",
            "Infertility",
            "Clinical Sign or Laboratory Abnormality",
            "NFC (Not Further Classified)",
            "Out of Scope: Acquired/Multifactorial",
        ],
        "allowed_tiers": ["1", "2", "3", "4", "NFC"],
        "allowed_severity_values": ["Severe", "Non-Severe"],
        "required_output_format": {
            "reasoning": {
                "preliminary_checks": {
                    "origin_type": "Inherited Genetic | Developmental Anomaly | Acquired/Multifactorial | Unknown",
                    "phenotype_type": "string",
                    "body_system": "string",
                    "onset": "string",
                },
                "phenotype_definition": {
                    "definition": "string",
                    "supporting_quote": "string with [Ref N]",
                },
                "functional_impact": {
                    "affected_domains": ["string"],
                    "domain_evidence": "string with [Ref N]",
                },
                "category_determination": {
                    "category": "one of allowed_categories",
                    "primary_evidence": "string with [Ref N]",
                    "rationale": "string",
                },
                "severity_assessment": {
                    "severity": "Severe or Non-Severe",
                    "functional_severity_evidence": "string with [Ref N]",
                },
                "final_classification": {
                    "tier": "1 | 2 | 3 | 4 | NFC",
                    "synthesis": "string",
                },
                "conclusion": "Brief evidence-cited summary using [Ref N] tags.",
            },
            "classification": {
                "tier": "<1-4 or NFC>",
                "category": "<one of allowed_categories>",
                "severity_assessment": "<Severe or Non-Severe>",
            },
            "management_profile": {
                "management_category": [
                    "Surgical Intervention | Pharmacological | Therapeutic Support | "
                    "Monitoring/Surveillance | Dietary Management | Assistive Devices | "
                    "Palliative Care | No Treatment Required"
                ],
                "management_summary": "string",
            },
            "cited_references": ["[Ref N: source-name - article-title]"],
        },
    }
    return json.dumps(prompt, indent=2)


# =============================================================================
# 6. ReAct AGENT LOOP
# =============================================================================

ACTION_ALIASES = {
    "get_hpo_context": AgentAction.GET_HPO_CONTEXT,
    "retrieve_hpo_context": AgentAction.GET_HPO_CONTEXT,
    "get_mesh_term": AgentAction.GET_MESH_TERM,
    "transform_query": AgentAction.TRANSFORM_QUERY,
    "search_pubmed": AgentAction.SEARCH_PUBMED,
    "classify": AgentAction.CLASSIFY,
    "final_classification": AgentAction.CLASSIFY,
    "insufficient_info": AgentAction.INSUFFICIENT_INFO,
}


def execute_action(action: AgentAction, context: ReActContext,
                   ontology: pronto.Ontology, session: requests.Session) -> str:
    if action == AgentAction.GET_HPO_CONTEXT:
        context.hpo_context = get_hpo_context(context.hpo_id, ontology)
        return f"Retrieved HPO context for {context.hpo_context.get('name', 'N/A')}"

    if action == AgentAction.GET_MESH_TERM:
        context.mesh_term_searched = True
        if not context.hpo_context:
            return "Need HPO context before MeSH lookup."
        context.mesh_term = get_mesh_term(
            context.hpo_context.get("name", ""),
            context.hpo_context.get("synonyms", []),
        )
        return f"MeSH term: {context.mesh_term or 'not found'}"

    if action == AgentAction.TRANSFORM_QUERY:
        context.queries_transformed = True
        if not context.hpo_context:
            return "Need HPO context before query transformation."
        context.transformed_queries = transform_query(
            session, context.hpo_context, context.mesh_term)
        return f"Generated {len(context.transformed_queries)} queries."

    if action == AgentAction.SEARCH_PUBMED:
        context.pubmed_searched = True
        if not context.hpo_context:
            return "Need HPO context before PubMed search."
        context.pubmed_articles = search_pubmed(
            context.hpo_context.get("name", ""),
            context.hpo_context.get("synonyms", []),
            context.mesh_term,
        )
        return f"Found {len(context.pubmed_articles)} PubMed articles."

    if action == AgentAction.CLASSIFY:
        if not context.hpo_context:
            return "Cannot classify without HPO context."
        prompt = create_classification_prompt(context)
        system = ("You are a clinical phenotyping expert. Every statement must "
                  "be supported by the provided sources, and citations [Ref N] "
                  "must reference real sources.")
        result = call_llm(session, prompt, system, temperature=0.0)
        if result:
            context.final_classification = result
            return "Classification produced."
        return "Classification failed (LLM error)."

    if action == AgentAction.INSUFFICIENT_INFO:
        context.final_classification = {
            "reasoning": "Agent reported insufficient information.",
            "classification": {
                "tier": "NFC",
                "category": "NFC (Not Further Classified)",
                "severity_assessment": "Unknown",
            },
            "management_profile": {
                "management_category": [],
                "management_summary": "N/A",
            },
            "cited_references": [],
        }
        return "Recorded insufficient information."

    return f"Unknown action: {action}"


def run_react_agent(hpo_id: str, ontology: pronto.Ontology,
                    session: requests.Session) -> ReActContext:
    print(f"\n>>> Processing {hpo_id}")
    context = ReActContext(hpo_id=hpo_id)

    for iteration in range(MAX_REACT_ITERATIONS):
        plan = call_llm(session, create_planning_prompt(context),
                        "You are a ReAct planning agent.",
                        temperature=REACT_TEMPERATURE)
        if not plan:
            print("  Planning step failed. Stopping.")
            break

        thought = plan.get("thought", "")
        action_name = (plan.get("action") or "").lower().strip()
        action_enum = ACTION_ALIASES.get(action_name)
        if action_enum is None:
            print(f"  Unknown action '{action_name}'. Re-planning.")
            continue

        observation = execute_action(action_enum, context, ontology, session)
        print(f"  Iter {iteration + 1}: {action_enum.value} -> {observation[:120]}")
        context.steps.append(ReActStep(
            iteration=iteration + 1,
            thought=thought,
            action=action_enum,
            observation=observation,
        ))

        if action_enum in (AgentAction.CLASSIFY, AgentAction.INSUFFICIENT_INFO):
            break

    return context


# =============================================================================
# 7. RESULT FLATTENING
# =============================================================================

def flatten_result(context: ReActContext) -> Dict[str, Any]:
    classification = (context.final_classification or {}).get("classification", {}) \
        if isinstance(context.final_classification, dict) else {}
    management = (context.final_classification or {}).get("management_profile", {}) \
        if isinstance(context.final_classification, dict) else {}
    reasoning = (context.final_classification or {}).get("reasoning", "") \
        if isinstance(context.final_classification, dict) else ""

    return {
        "hpo_id": context.hpo_id,
        "hpo_name": (context.hpo_context or {}).get("name", "N/A"),
        "mesh_term": context.mesh_term or "Not Found",
        "tier": classification.get("tier", "NFC"),
        "category": classification.get("category", "N/A"),
        "severity_assessment": classification.get("severity_assessment", "N/A"),
        "management_category": json.dumps(management.get("management_category", [])),
        "management_summary": management.get("management_summary", "N/A"),
        "pubmed_results_count": len(context.pubmed_articles),
        "react_steps": len(context.steps),
        "reasoning": json.dumps(reasoning) if isinstance(reasoning, dict)
                     else str(reasoning),
        "references": json.dumps(context.reference_urls),
    }


# =============================================================================
# 8. MAIN
# =============================================================================

def main() -> int:
    print("=" * 70)
    print("HPO Term Classification ReAct Agent (Reproducibility Build)")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Data dir: {DATA_DIR}")
    print(f"Results dir: {RESULTS_DIR}")

    if not OPENROUTER_API_KEY:
        print("\nERROR: OPENROUTER_API_KEY environment variable is not set.")
        print("Set it as a CodeOcean Secret (or export it locally) and re-run.")
        return 1

    ontology = load_hpo_ontology(HPO_ONTOLOGY_PATH)
    if ontology is None:
        return 1

    if not os.path.exists(INPUT_CSV_PATH):
        print(f"ERROR: Input file not found at {INPUT_CSV_PATH}")
        return 1

    df = pd.read_csv(INPUT_CSV_PATH)
    if "hpo_id" not in df.columns:
        print("ERROR: Input CSV must contain a 'hpo_id' column.")
        return 1
    hpo_ids = df["hpo_id"].dropna().unique().tolist()
    print(f"Loaded {len(hpo_ids)} HPO terms from {INPUT_CSV_PATH}")

    flattened_rows: List[Dict[str, Any]] = []
    full_traces: List[Dict[str, Any]] = []
    start = time.time()

    with requests.Session() as session:
        for hpo_id in hpo_ids:
            try:
                context = run_react_agent(hpo_id, ontology, session)
                flattened_rows.append(flatten_result(context))
                full_traces.append({
                    "hpo_id": hpo_id,
                    "hpo_context": context.hpo_context,
                    "mesh_term": context.mesh_term,
                    "transformed_queries": context.transformed_queries,
                    "pubmed_articles": [
                        {k: a.get(k) for k in ("source", "id", "url", "title")}
                        for a in context.pubmed_articles
                    ],
                    "steps": [
                        {"iteration": s.iteration, "thought": s.thought,
                         "action": s.action.value, "observation": s.observation}
                        for s in context.steps
                    ],
                    "final_classification": context.final_classification,
                })
            except Exception as e:
                print(f"  ERROR while processing {hpo_id}: {e}")
                flattened_rows.append({
                    "hpo_id": hpo_id, "hpo_name": "ERROR",
                    "mesh_term": "", "tier": "ERROR", "category": str(e),
                    "severity_assessment": "ERROR",
                    "management_category": "[]", "management_summary": "",
                    "pubmed_results_count": 0, "react_steps": 0,
                    "reasoning": "", "references": "{}",
                })

    pd.DataFrame(flattened_rows).to_csv(OUTPUT_CSV_PATH, index=False)
    with open(REASONING_LOG_PATH, "w") as f:
        json.dump(full_traces, f, indent=2)

    elapsed = (time.time() - start) / 60.0
    print("\n" + "=" * 70)
    print(f"Done in {elapsed:.2f} minutes")
    print(f"  Classifications: {OUTPUT_CSV_PATH}")
    print(f"  Reasoning log:   {REASONING_LOG_PATH}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
