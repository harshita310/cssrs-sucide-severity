# Neo4j XAI Clinical Decision Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Neo4j-first post-classification pipeline that explains final focused CE MentalBERT predictions with SHAP and retrieves evidence-backed interventions through graph traversal.

**Architecture:** Keep the trained classifier unchanged and add a separate `knowledge_graph/` package plus scripts. Neo4j is the required backend for graph population and traversal; local files are seed data only. Reports are deterministic JSON/Markdown outputs generated from prediction probabilities, SHAP token factors, Neo4j concepts, interventions, evidence, and resources.

**Tech Stack:** Python, PyTorch, Transformers, SHAP, Neo4j Python Driver, pandas, matplotlib, pytest.

## Global Constraints

- Do not retrain or modify the final focused CE MentalBERT model.
- Use Neo4j as the graph backend; do not implement a local recommendation fallback.
- Do not generate free-form recommendations with an LLM.
- Store Neo4j passwords only in environment variables.
- Store SHAP outputs under `RESULTS/xai/shap/`.
- Store report outputs under `RESULTS/xai/reports/`.
- Keep the ordinal model retained as comparison context, not as the main XAI model.
- Include a research-only disclaimer in generated clinical reports.

---

## File Structure

- Create `knowledge_graph/__init__.py`: public package exports.
- Create `knowledge_graph/schema.py`: dataclasses for seed nodes, relationships, SHAP factors, recommendations, and reports.
- Create `knowledge_graph/text.py`: normalization helpers for aliases and tokens.
- Create `knowledge_graph/graph_loader.py`: Neo4j connection, environment loading, constraints.
- Create `knowledge_graph/graph_builder.py`: seed-data validation and Neo4j population.
- Create `knowledge_graph/graph_query.py`: Cypher traversal functions.
- Create `knowledge_graph/shap_mapper.py`: SHAP token to Neo4j concept mapping.
- Create `knowledge_graph/retriever.py`: evidence/resource retrieval.
- Create `knowledge_graph/recommendation_engine.py`: deterministic intervention ranking.
- Create `knowledge_graph/clinical_report.py`: JSON/Markdown report assembly.
- Create `knowledge_graph/seed_data.py`: curated minimal clinical graph seed data.
- Create `xai/__init__.py` and `xai/shap_explainer.py`: model loading, prediction, SHAP explanation, artifact writing.
- Create `scripts/build_knowledge_graph.py`: build Neo4j constraints and seed graph.
- Create `scripts/run_shap_explanation.py`: produce SHAP output for one text.
- Create `scripts/generate_clinical_report.py`: full post-classification report flow.
- Modify `configs/default.yaml`: add Neo4j and XAI settings.
- Modify `requirements.txt`: add `neo4j` and `shap`.
- Modify `README.md`: document setup and scripts.
- Add tests under `tests/` for normalization, seed validation, query text, ranking, report shape, and SHAP artifact formatting.

---

### Task 1: Configuration, Dependencies, and Core Schemas

**Files:**
- Modify: `requirements.txt`
- Modify: `configs/default.yaml`
- Create: `knowledge_graph/__init__.py`
- Create: `knowledge_graph/schema.py`
- Create: `knowledge_graph/text.py`
- Test: `tests/test_knowledge_graph_schema.py`

**Interfaces:**
- Produces: `normalize_key(text: str) -> str`
- Produces: `ShapTokenFactor(token: str, value: float, rank: int, direction: str)`
- Produces: `MappedConcept(name: str, label: str, matched_alias: str, shap_value: float)`
- Produces: `Recommendation(name: str, score: float, concepts: list[str], evidence: list[dict], resources: list[dict])`

- [ ] **Step 1: Write schema and normalization tests**

```python
from knowledge_graph.schema import MappedConcept, Recommendation, ShapTokenFactor
from knowledge_graph.text import normalize_key


def test_normalize_key_lowercases_and_removes_noise():
    assert normalize_key(" Can't sleep!! ") == "cant sleep"


def test_shap_token_factor_requires_direction():
    factor = ShapTokenFactor(token="hopeless", value=0.42, rank=1, direction="positive")
    assert factor.direction == "positive"


def test_recommendation_schema_holds_traceable_evidence():
    rec = Recommendation(
        name="Safety Planning",
        score=2.5,
        concepts=["Hopelessness"],
        evidence=[{"name": "WHO guideline", "passage": "Safety planning is evidence linked."}],
        resources=[{"name": "Emergency services"}],
    )
    assert rec.evidence[0]["name"] == "WHO guideline"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests\test_knowledge_graph_schema.py -q`

Expected: FAIL because `knowledge_graph` does not exist.

- [ ] **Step 3: Add dependencies and config**

Add to `requirements.txt`:

```text
# Explainability and graph backend
shap>=0.45.0,<1.0.0
neo4j>=5.20.0,<6.0.0
```

Add to `configs/default.yaml`:

```yaml
neo4j:
  uri: "bolt://localhost:7687"
  username: "neo4j"
  password_env: "NEO4J_PASSWORD"
  database: "neo4j"

xai:
  shap_output_dir: "RESULTS/xai/shap"
  reports_output_dir: "RESULTS/xai/reports"
  top_k_tokens: 12
  top_k_recommendations: 5
```

- [ ] **Step 4: Implement schemas and normalization**

```python
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ShapTokenFactor:
    token: str
    value: float
    rank: int
    direction: str


@dataclass(frozen=True)
class MappedConcept:
    name: str
    label: str
    matched_alias: str
    shap_value: float


@dataclass(frozen=True)
class Recommendation:
    name: str
    score: float
    concepts: list[str] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    resources: list[dict] = field(default_factory=list)
```

```python
import re
import unicodedata


def normalize_key(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower().strip()
    normalized = re.sub(r"[^a-z0-9\s]+", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests\test_knowledge_graph_schema.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt configs/default.yaml knowledge_graph tests/test_knowledge_graph_schema.py
git commit -m "Add Neo4j XAI schemas and config"
```

---

### Task 2: Neo4j Loader and Seed Graph Builder

**Files:**
- Create: `knowledge_graph/graph_loader.py`
- Create: `knowledge_graph/seed_data.py`
- Create: `knowledge_graph/graph_builder.py`
- Create: `scripts/build_knowledge_graph.py`
- Test: `tests/test_graph_builder.py`

**Interfaces:**
- Consumes: `normalize_key(text: str) -> str`
- Produces: `Neo4jSettings.from_config(cfg) -> Neo4jSettings`
- Produces: `build_seed_statements() -> list[tuple[str, dict]]`
- Produces: `create_constraints(driver, database: str) -> None`
- Produces: `load_seed_graph(driver, database: str) -> None`

- [ ] **Step 1: Write tests for seed validation and Cypher generation**

```python
from knowledge_graph.graph_builder import build_seed_statements
from knowledge_graph.seed_data import SEED_CONCEPTS, SEED_INTERVENTIONS


def test_seed_data_contains_required_clinical_concepts():
    names = {item["name"] for item in SEED_CONCEPTS}
    assert {"Hopelessness", "Worthlessness", "Isolation", "Insomnia", "Self Harm"} <= names


def test_build_seed_statements_returns_parameterized_cypher():
    statements = build_seed_statements()
    assert statements
    assert all("$" in cypher for cypher, _ in statements)
    assert any("MERGE (n:Intervention" in cypher for cypher, _ in statements)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests\test_graph_builder.py -q`

Expected: FAIL because builder files do not exist.

- [ ] **Step 3: Create seed data**

Include concepts with aliases:

```python
SEED_CONCEPTS = [
    {"label": "Symptom", "name": "Hopelessness", "aliases": ["hopeless", "no hope", "hopelessness"]},
    {"label": "Symptom", "name": "Worthlessness", "aliases": ["worthless", "burden", "useless"]},
    {"label": "Symptom", "name": "Isolation", "aliases": ["alone", "lonely", "isolated"]},
    {"label": "Symptom", "name": "Insomnia", "aliases": ["cant sleep", "cannot sleep", "sleepless", "insomnia"]},
    {"label": "Symptom", "name": "Self Harm", "aliases": ["self harm", "cut myself", "hurt myself"]},
    {"label": "Emotion", "name": "Sadness", "aliases": ["sad", "sadness", "depressed"]},
    {"label": "Emotion", "name": "Anxiety", "aliases": ["anxious", "panic", "afraid"]},
]
```

Include interventions, evidence, resources, and relationships linking severity bands to interventions.

- [ ] **Step 4: Implement Neo4j loader**

Use `neo4j.GraphDatabase.driver(uri, auth=(username, password))`, reading the password from `os.environ[password_env]`. Raise `RuntimeError("Missing Neo4j password environment variable: NEO4J_PASSWORD")` when absent.

- [ ] **Step 5: Implement graph builder**

Generate parameterized `MERGE` statements for:

```cypher
MERGE (n:Symptom {name: $name})
MERGE (a:Alias {key: $key})
MERGE (a)-[:ALIAS_OF]->(n)
MERGE (i:Intervention {name: $name})
MERGE (e:EvidenceSource {name: $name})
MERGE (i)-[:SUPPORTED_BY]->(e)
MERGE (s:SeverityBand {level: $level})
MERGE (s)-[:RECOMMENDS]->(i)
```

- [ ] **Step 6: Add build script**

`scripts/build_knowledge_graph.py` loads config, opens Neo4j, creates constraints, loads seed graph, and prints a summary count.

- [ ] **Step 7: Run tests**

Run: `.venv\Scripts\python.exe -m pytest tests\test_graph_builder.py -q`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add knowledge_graph/graph_loader.py knowledge_graph/seed_data.py knowledge_graph/graph_builder.py scripts/build_knowledge_graph.py tests/test_graph_builder.py
git commit -m "Add Neo4j seed graph builder"
```

---

### Task 3: Graph Query, SHAP Mapping, Evidence Retrieval, and Ranking

**Files:**
- Create: `knowledge_graph/graph_query.py`
- Create: `knowledge_graph/shap_mapper.py`
- Create: `knowledge_graph/retriever.py`
- Create: `knowledge_graph/recommendation_engine.py`
- Test: `tests/test_graph_query_and_recommendations.py`

**Interfaces:**
- Consumes: `ShapTokenFactor`, `MappedConcept`, `Recommendation`
- Produces: `concept_query() -> str`
- Produces: `intervention_query() -> str`
- Produces: `map_shap_tokens(query, factors: list[ShapTokenFactor]) -> list[MappedConcept]`
- Produces: `rank_recommendations(rows: list[dict], top_k: int) -> list[Recommendation]`

- [ ] **Step 1: Write tests for Cypher and deterministic ranking**

```python
from knowledge_graph.graph_query import concept_query, intervention_query
from knowledge_graph.recommendation_engine import rank_recommendations


def test_concept_query_uses_alias_lookup():
    cypher = concept_query()
    assert "Alias" in cypher
    assert "ALIAS_OF" in cypher
    assert "$keys" in cypher


def test_intervention_query_uses_concepts_and_severity():
    cypher = intervention_query()
    assert "SUPPORTED_BY" in cypher
    assert "SeverityBand" in cypher
    assert "$concept_names" in cypher
    assert "$severity" in cypher


def test_rank_recommendations_prefers_more_evidence_and_higher_shap():
    rows = [
        {"intervention": "Peer Support", "concept": "Isolation", "shap_value": 0.4, "evidence": {"name": "Evidence A"}, "resource": {"name": "Group"}},
        {"intervention": "Safety Planning", "concept": "Hopelessness", "shap_value": 0.9, "evidence": {"name": "Evidence B"}, "resource": {"name": "Professional"}},
        {"intervention": "Safety Planning", "concept": "Self Harm", "shap_value": 0.7, "evidence": {"name": "Evidence C"}, "resource": {"name": "Helpline"}},
    ]
    ranked = rank_recommendations(rows, top_k=2)
    assert ranked[0].name == "Safety Planning"
    assert ranked[0].score > ranked[1].score
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests\test_graph_query_and_recommendations.py -q`

Expected: FAIL because query/ranking modules do not exist.

- [ ] **Step 3: Implement Cypher query builders**

`concept_query()` returns alias-to-concept lookup:

```cypher
MATCH (a:Alias)-[:ALIAS_OF]->(c)
WHERE a.key IN $keys
RETURN a.key AS alias_key, c.name AS concept_name, labels(c)[0] AS concept_label
```

`intervention_query()` returns concept and severity linked interventions with evidence/resources.

- [ ] **Step 4: Implement SHAP mapper**

Normalize token factors into alias keys, run `concept_query()`, and return `MappedConcept` objects with the original SHAP value preserved.

- [ ] **Step 5: Implement deterministic ranking**

Group rows by intervention. Score each intervention as:

```text
sum(abs(shap_value) for linked concepts) + 0.25 * evidence_count + 0.15 * resource_count
```

Deduplicate evidence/resources by `name`. Sort by `score` descending, then intervention name ascending.

- [ ] **Step 6: Run tests**

Run: `.venv\Scripts\python.exe -m pytest tests\test_graph_query_and_recommendations.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add knowledge_graph/graph_query.py knowledge_graph/shap_mapper.py knowledge_graph/retriever.py knowledge_graph/recommendation_engine.py tests/test_graph_query_and_recommendations.py
git commit -m "Add Neo4j graph traversal and recommendation ranking"
```

---

### Task 4: SHAP Explainer for Final Focused CE Model

**Files:**
- Create: `xai/__init__.py`
- Create: `xai/shap_explainer.py`
- Create: `scripts/run_shap_explanation.py`
- Test: `tests/test_shap_explainer_outputs.py`

**Interfaces:**
- Produces: `PredictionResult(label: int, confidence: float, probabilities: list[float])`
- Produces: `ShapExplanation(prediction: PredictionResult, positive: list[ShapTokenFactor], negative: list[ShapTokenFactor], values: list[dict])`
- Produces: `write_shap_artifacts(explanation: ShapExplanation, output_dir: Path, run_id: str) -> dict[str, Path]`

- [ ] **Step 1: Write tests for artifact formatting**

```python
import json
from pathlib import Path

from knowledge_graph.schema import ShapTokenFactor
from xai.shap_explainer import PredictionResult, ShapExplanation, write_shap_artifacts


def test_write_shap_artifacts_creates_json_and_csv(tmp_path: Path):
    explanation = ShapExplanation(
        prediction=PredictionResult(label=3, confidence=0.91, probabilities=[0.01, 0.02, 0.03, 0.91, 0.01, 0.01, 0.01]),
        positive=[ShapTokenFactor("hopeless", 0.6, 1, "positive")],
        negative=[ShapTokenFactor("support", -0.2, 1, "negative")],
        values=[{"token": "hopeless", "value": 0.6}],
    )
    paths = write_shap_artifacts(explanation, tmp_path, "sample")
    assert paths["json"].exists()
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["prediction"]["label"] == 3
    assert paths["tokens_csv"].exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests\test_shap_explainer_outputs.py -q`

Expected: FAIL because `xai/shap_explainer.py` does not exist.

- [ ] **Step 3: Implement model prediction wrapper**

Load `cfg.MODEL_PATH`, tokenizer, and MentalBERT with existing utilities. Predict by softmax, returning label, confidence, and probabilities.

- [ ] **Step 4: Implement SHAP explanation**

Use `shap.Explainer` with a callable that accepts text strings and returns probability arrays. Tokenize through the existing MentalBERT tokenizer and run on CUDA when available. Extract top positive and negative token factors for the predicted class.

- [ ] **Step 5: Implement artifact writer and CLI**

Write JSON and CSV to `RESULTS/xai/shap/`. The CLI supports:

```powershell
.venv\Scripts\python.exe scripts\run_shap_explanation.py --text "I feel hopeless and alone"
```

- [ ] **Step 6: Run tests**

Run: `.venv\Scripts\python.exe -m pytest tests\test_shap_explainer_outputs.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add xai scripts/run_shap_explanation.py tests/test_shap_explainer_outputs.py
git commit -m "Add SHAP explainer artifacts"
```

---

### Task 5: Clinical Report Generator

**Files:**
- Create: `knowledge_graph/clinical_report.py`
- Create: `scripts/generate_clinical_report.py`
- Test: `tests/test_clinical_report.py`

**Interfaces:**
- Consumes: `PredictionResult`, `ShapExplanation`, `MappedConcept`, `Recommendation`
- Produces: `build_report(...) -> dict`
- Produces: `write_report(report: dict, output_dir: Path, run_id: str) -> dict[str, Path]`

- [ ] **Step 1: Write report tests**

```python
import json
from pathlib import Path

from knowledge_graph.clinical_report import build_report, write_report
from knowledge_graph.schema import MappedConcept, Recommendation, ShapTokenFactor
from xai.shap_explainer import PredictionResult


def test_build_report_contains_traceability_and_disclaimer():
    report = build_report(
        input_preview="I feel hopeless",
        prediction=PredictionResult(label=4, confidence=0.88, probabilities=[0, 0, 0, 0, 0.88, 0.1, 0.02]),
        positive_factors=[ShapTokenFactor("hopeless", 0.7, 1, "positive")],
        negative_factors=[],
        concepts=[MappedConcept("Hopelessness", "Symptom", "hopeless", 0.7)],
        recommendations=[Recommendation("Safety Planning", 1.2, ["Hopelessness"], [{"name": "WHO"}], [{"name": "Helpline"}])],
    )
    assert report["prediction"]["label"] == 4
    assert report["recommendations"][0]["evidence"][0]["name"] == "WHO"
    assert "research" in report["disclaimer"].lower()


def test_write_report_creates_json_and_markdown(tmp_path: Path):
    report = {"prediction": {"label": 2}, "recommendations": [], "disclaimer": "research only"}
    paths = write_report(report, tmp_path, "sample")
    assert json.loads(paths["json"].read_text(encoding="utf-8"))["prediction"]["label"] == 2
    assert paths["markdown"].exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests\test_clinical_report.py -q`

Expected: FAIL because report module does not exist.

- [ ] **Step 3: Implement deterministic report builder**

Report keys:

```python
{
  "input_preview": "...",
  "prediction": {"label": 4, "confidence": 0.88, "probabilities": [...]},
  "shap": {"positive_factors": [...], "negative_factors": [...]},
  "mapped_concepts": [...],
  "recommendations": [...],
  "disclaimer": "Research decision-support output only; not a diagnosis or emergency response tool."
}
```

- [ ] **Step 4: Implement full CLI flow**

`scripts/generate_clinical_report.py`:

1. Run prediction and SHAP explanation.
2. Connect to Neo4j.
3. Map SHAP tokens to concepts.
4. Query interventions/evidence/resources.
5. Rank recommendations.
6. Write JSON/Markdown report.

- [ ] **Step 5: Run tests**

Run: `.venv\Scripts\python.exe -m pytest tests\test_clinical_report.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add knowledge_graph/clinical_report.py scripts/generate_clinical_report.py tests/test_clinical_report.py
git commit -m "Add deterministic clinical report generator"
```

---

### Task 6: Documentation and End-to-End Verification

**Files:**
- Modify: `README.md`
- Test: all focused XAI unit tests

**Interfaces:**
- Consumes all previous scripts and modules.
- Produces documented commands for Neo4j setup, graph build, SHAP explanation, and report generation.

- [ ] **Step 1: Add README section**

Document:

```powershell
$env:NEO4J_PASSWORD="your-password"
.venv\Scripts\python.exe scripts\build_knowledge_graph.py
.venv\Scripts\python.exe scripts\run_shap_explanation.py --text "I feel hopeless and alone"
.venv\Scripts\python.exe scripts\generate_clinical_report.py --text "I feel hopeless and alone"
```

Explain that Neo4j must be running at `bolt://localhost:7687`.

- [ ] **Step 2: Run full targeted test suite**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_knowledge_graph_schema.py tests\test_graph_builder.py tests\test_graph_query_and_recommendations.py tests\test_shap_explainer_outputs.py tests\test_clinical_report.py -q
```

Expected: PASS.

- [ ] **Step 3: Run compile check**

Run:

```powershell
.venv\Scripts\python.exe -m py_compile knowledge_graph\schema.py knowledge_graph\text.py knowledge_graph\graph_loader.py knowledge_graph\graph_builder.py knowledge_graph\graph_query.py knowledge_graph\shap_mapper.py knowledge_graph\retriever.py knowledge_graph\recommendation_engine.py knowledge_graph\clinical_report.py xai\shap_explainer.py scripts\build_knowledge_graph.py scripts\run_shap_explanation.py scripts\generate_clinical_report.py
```

Expected: exit code 0.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "Document Neo4j XAI clinical support workflow"
```

---

## Self-Review

- Spec coverage: The plan covers SHAP integration, Neo4j schema, graph construction, population script, SHAP-to-graph mapping, graph traversal, evidence retrieval, recommendation ranking, clinical report generation, and documentation.
- Placeholder scan: No TBD/TODO placeholders are used as implementation requirements.
- Type consistency: `ShapTokenFactor`, `MappedConcept`, `Recommendation`, `PredictionResult`, and `ShapExplanation` are introduced before later tasks consume them.
