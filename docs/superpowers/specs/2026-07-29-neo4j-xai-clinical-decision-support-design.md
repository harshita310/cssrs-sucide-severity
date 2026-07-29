# Neo4j XAI Clinical Decision Support Design

## Goal

Extend the trained MentalBERT-CSSR classifier into a deterministic,
explainable, evidence-grounded clinical decision support research framework.
The final focused CE checkpoint remains unchanged and is used only for
prediction. No model retraining is part of this extension.

## Final Data Flow

```text
User text
  -> focused CE MentalBERT prediction
  -> SHAP token explanation
  -> top token/concept extraction
  -> Neo4j concept mapping and graph traversal
  -> evidence-backed intervention retrieval
  -> structured clinical report
```

## Scope

This implementation is Neo4j-first. The system requires a running Neo4j
database for graph population and traversal. Local JSON files may be used only
as seed data for loading Neo4j, not as a replacement recommendation engine.

The system does not generate free-form clinical advice. It returns structured
recommendations by retrieving graph-linked interventions, evidence sources, and
resources. Text in the report comes from templates and stored evidence fields.

## Components

### SHAP Explanation

Add a module that loads the final focused CE checkpoint, predicts severity, and
computes token-level SHAP values for the predicted class and optionally for all
classes. The output includes prediction label, confidence, top positive tokens,
top negative tokens, raw SHAP values, and saved visualizations.

Outputs are written under `RESULTS/xai/shap/` as JSON/CSV/PNG artifacts. These
outputs become the input to graph mapping.

### Neo4j Knowledge Graph

Add `knowledge_graph/` modules:

- `graph_loader.py`: connection handling and constraint creation.
- `graph_builder.py`: seed symptoms, emotions, interventions, evidence, and
  resources into Neo4j.
- `graph_query.py`: reusable Cypher traversal functions.
- `shap_mapper.py`: map SHAP tokens to graph concepts through aliases and graph
  lookup queries.
- `retriever.py`: retrieve linked evidence nodes and passages.
- `recommendation_engine.py`: combine severity prediction and mapped concepts
  to rank interventions.
- `clinical_report.py`: build deterministic JSON/Markdown reports.

### Graph Schema

Node labels:

- `Symptom`
- `Emotion`
- `Intervention`
- `EvidenceSource`
- `Resource`
- `SeverityBand`
- `Alias`

Relationships:

- `(Alias)-[:ALIAS_OF]->(Symptom|Emotion)`
- `(Symptom|Emotion)-[:TREATED_BY]->(Intervention)`
- `(Symptom|Emotion)-[:BENEFITS_FROM]->(Intervention|Resource)`
- `(Intervention)-[:SUPPORTED_BY]->(EvidenceSource)`
- `(SeverityBand)-[:RECOMMENDS]->(Intervention|Resource)`
- `(EvidenceSource)-[:CITES]->(EvidenceSource)` where useful

The graph stores traceable fields such as `name`, `description`, `source_type`,
`citation`, `url`, `passage`, and `risk_level`.

### SHAP to Graph Mapping

Token mapping avoids long if-else chains. Each extracted token is normalized and
queried against `Alias` nodes. Matched aliases point to clinical concepts such
as `Hopelessness`, `Worthlessness`, `Isolation`, `Insomnia`, and `Self Harm`.

Semantic matching with SentenceTransformers can be added later, but the first
implementation uses curated aliases in Neo4j so results are reproducible and
easy to defend academically.

### Recommendation Logic

Recommendations depend on both:

1. MentalBERT predicted severity band.
2. SHAP-derived graph concepts.

The engine ranks interventions using deterministic factors:

- number of mapped SHAP concepts linked to the intervention
- SHAP contribution strength for those concepts
- severity-band links
- evidence availability

The engine returns interventions only when they are connected to evidence nodes.

### Clinical Report

The report includes:

- input text identifier or preview
- predicted severity and confidence
- top SHAP positive and negative factors
- mapped graph concepts
- recommended interventions
- evidence sources and passages
- resources
- clear research-only disclaimer

Reports are stored under `RESULTS/xai/reports/` as JSON and Markdown.

## Configuration

Add Neo4j settings to `configs/default.yaml`:

- URI
- username
- password environment-variable name
- database

The password is never committed. Scripts read it from the environment.

## Scripts

Add command-line entry points:

- `scripts/build_knowledge_graph.py`
- `scripts/run_shap_explanation.py`
- `scripts/generate_clinical_report.py`

The first script creates constraints and loads seed data into Neo4j. The report
script performs the full post-classification flow for one text input or a CSV
of examples.

## Testing

Unit tests cover normalization, seed-data validation, Cypher query generation,
recommendation ranking, and report structure. Neo4j integration tests are
optional and skipped unless Neo4j connection environment variables are present.

## Out of Scope

- retraining MentalBERT
- merging labels
- replacing Neo4j with local-only rules
- free-form LLM recommendation generation
- clinical diagnosis or emergency triage automation

## Research Contribution

The contribution is a deterministic XAI pipeline where MentalBERT predicts
severity, SHAP explains influential text evidence, Neo4j maps those explanations
to clinical concepts, graph traversal retrieves evidence-backed interventions,
and a structured report preserves traceability from prediction to recommendation.
