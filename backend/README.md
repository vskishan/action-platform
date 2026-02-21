# ACTION Platform — Backend

> **A**I-powered **C**linical **T**rial **I**ntelligence and **O**rchestration **N**etwork

## Overview

The backend is a **FastAPI** application that powers the ACTION clinical-trial
research platform. It combines:

- **Federated Patient Screening** — MedGemma-powered eligibility screening
  across distributed hospital sites (no patient-level data leaves any site).
- **Cohort Formation** — A ReAct (Reason + Act) agent that autonomously queries
  survival-analysis and analytics engines to answer researcher questions.
- **Cohort Monitoring** — Federated treatment-arm progress monitoring with
  MedGemma-driven clinical-note extraction.
- **Workflow Orchestration** — An intelligent, AI-driven workflow engine that
  guides trials through screening → cohort formation → monitoring.

## Architecture

```
backend/
├── app/
│   ├── main.py                  # FastAPI entry point & lifespan
│   ├── config/                  # (reserved for future configuration)
│   ├── engine/                  # Domain engines & state management
│   │   ├── base_engine.py       # Abstract engine interface
│   │   ├── analytics_engine.py  # Descriptive-analytics engine
│   │   ├── survival_engine.py   # Cox PH survival-analysis engine
│   │   ├── workflow_engine.py   # Workflow state-machine logic
│   │   ├── workflow_store.py    # In-memory workflow persistence
│   │   └── job_store.py         # Background-job subsystem
│   ├── federated/               # Flower-based federated learning
│   │   ├── central_server.py    # Screening orchestrator (Flower server)
│   │   ├── federated_client.py  # Per-site screening client
│   │   ├── monitoring_server.py # Monitoring orchestrator (Flower server)
│   │   └── monitoring_client.py # Per-site monitoring client
│   ├── llm/                     # LLM / AI agent layer
│   │   ├── medgemma_client.py   # Ollama MedGemma wrapper (singleton)
│   │   ├── agent.py             # Simple intent-classify → engine agent
│   │   ├── react_agent.py       # ReAct agentic orchestrator
│   │   ├── intent_classifier.py # Query → engine/intent classifier
│   │   ├── screening_auditor.py # Self-correcting screening agent
│   │   ├── workflow_orchestrator.py # Autonomous workflow intelligence
│   │   ├── tools.py             # Tool registry for the ReAct agent
│   │   ├── memory.py            # Session-based conversation memory
│   │   └── prompts.py           # Centralised LLM prompt templates
│   ├── model/                   # Serialised ML model artefacts
│   ├── routes/                  # FastAPI route modules
│   │   ├── cohort/              # Cohort formation endpoints
│   │   ├── federated/           # Screening & monitoring endpoints
│   │   └── workflow/            # Workflow CRUD & job endpoints
│   └── schema/                  # Pydantic request/response models
│       ├── analytics_schema.py
│       ├── monitoring_schema.py
│       ├── prediction_schema.py
│       ├── screening_schema.py
│       └── workflow_schema.py
├── tests/                       # Pytest test suite
└── README.md                    # ← you are here
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Ensure Ollama is running with the MedGemma model
ollama pull alibayram/medgemma

# 3. Start the development server
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **In-memory stores** | For simulation purposes; we'll be using a real DB in production |
| **Singleton patterns** | MedGemma client, workflow store, and job store are singletons to share state across requests |
| **Flower framework** | Provides a production-grade federated learning protocol; only aggregate counts cross site boundaries |
| **ReAct agent** | Enables autonomous, multi-step reasoning over survival and analytics data |
| **Self-correcting screening** | Two-pass screen → audit → reflect pipeline improves accuracy without human review |

## Running Tests

```bash
pytest backend/tests/ -v
```
