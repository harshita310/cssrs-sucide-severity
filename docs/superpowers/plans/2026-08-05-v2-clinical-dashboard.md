# V2 Clinical Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a V2 clinical research dashboard around the existing MentalBERT, SHAP, Neo4j, and evidence retrieval pipeline without changing the V1 report workflow.

**Architecture:** Add a Python API layer that converts prediction, explanation, concept mapping, graph traversal, and evidence chunks into a deterministic dashboard JSON payload. Add a separate React/Vite dashboard that presents prediction, SHAP factors, evidence cards, graph trace, recommendation explorer, and report export.

**Tech Stack:** Python, FastAPI, Neo4j Python driver, existing MentalBERT/SHAP modules, React, Vite, TailwindCSS, Framer Motion, Lucide, Recharts, React Flow.

## Global Constraints

- Keep previous V1 scripts and report generation behavior intact.
- Do not use GPT, Gemini, Claude, or any LLM after prediction.
- Every clinical statement shown in V2 must come from retrieved graph/evidence data or static model metadata.
- Keep generated results and large model artifacts out of git.

---

### Task 1: Backend Dashboard Payload

**Files:**
- Create: `api/__init__.py`
- Create: `api/schemas.py`
- Create: `api/dashboard_service.py`
- Create: `api/app.py`
- Test: `tests/test_dashboard_service.py`
- Modify: `requirements.txt`

**Interfaces:**
- Produces: `build_dashboard_payload(text, explanation, concepts, recommendations) -> dict`
- Produces: FastAPI app with `/api/health`, `/api/analyze`, `/api/report/html`

- [ ] Write failing tests for prediction, SHAP, concept, evidence, graph, and export payload fields.
- [ ] Run the tests and confirm they fail because `api.dashboard_service` does not exist.
- [ ] Implement the minimal deterministic payload builder.
- [ ] Add FastAPI endpoints that call existing predictor, SHAP, Neo4j mapper, retriever, and report writer.
- [ ] Run targeted tests and full existing tests.

### Task 2: React Dashboard Scaffold

**Files:**
- Create: `dashboard/package.json`
- Create: `dashboard/index.html`
- Create: `dashboard/src/main.jsx`
- Create: `dashboard/src/App.jsx`
- Create: `dashboard/src/styles.css`
- Create: `dashboard/src/api.js`
- Create: `dashboard/src/components/*.jsx`

**Interfaces:**
- Consumes: `/api/analyze` dashboard payload.
- Produces: V2 dashboard UI with left prediction panel, center evidence/explainability panel, and right graph panel.

- [ ] Create the Vite app structure without disturbing Python code.
- [ ] Build polished responsive panels, token chips, evidence cards, recommendation cards, and graph trace.
- [ ] Add dark/light mode and export controls.
- [ ] Add sample fallback payload for UI preview if backend is not running.

### Task 3: Documentation, Screenshot, and GitHub

**Files:**
- Modify: `README.md`
- Create: `docs/assets/v2_dashboard_placeholder.md` or replace with screenshot when available.

- [ ] Document backend setup, frontend setup, Neo4j requirement, and no-LLM guarantee.
- [ ] Run backend tests.
- [ ] Run frontend build if Node dependencies are available.
- [ ] Capture or prepare dashboard screenshot instructions.
- [ ] Commit and push source changes to GitHub.
