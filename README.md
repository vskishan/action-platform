## Project name 
**ACTION** : **A**I-Powered **C**linical **T**rial platform **I**mproved and **O**rchestrated using **N**ext-Gen workflows

## Problem statement
#### The Human Cost of Slow Clinical Trials

Clinical trials are the final bridge between a scientific breakthrough and a life-saving treatment. Before any therapy reaches patients, it must pass through a multi-phase process evaluating safety and effectiveness.

In 2024 alone, nearly 2 million Americans were diagnosed with cancer. For many, the most promising treatments exist only inside clinical trials.

Yet the system designed to deliver those therapies is slow, fragmented, and inefficient.

Behind every delay is someone waiting for a chance at survival.

#### What Happens Today
Consider the journey of a single oncology trial today:

1. A **research coordinator** spends **weeks** manually screening EHRs against multi-page eligibility criteria written in complex medical language with **~85%** of patients ultimately ineligible.
2. A **biostatistician** waits **months** to form a statistically viable cohort while relevant control data remains siloed across institutions.
3. A **principal investigator** learns of serious adverse events days after they occur due to manual reporting pipelines.

This is not an exception. This has been a practice till now.

####The Scale of the Problem

| Metric | Reality |
|---|---|
| Average trial duration | **7+ years** from Phase I to approval |
| Average cost per approved drug | **$100 million+** |
| Patient screening failure rate | **~85 %** |
| Trials terminated for slow enrollment | **~30 %**|

<br>

#### Why AI and Why MedGemma
Clinical trials are inherently:
- Language-heavy (eligibility criteria, physician notes)
- Data-rich (labs, imaging, longitudinal records)
- Decision-intensive (eligibility, safety grading, cohort analytics)

Medically specialized models like MedGemma are uniquely suited because they combine:
- Clinical language understanding
- Structured + unstructured health data interpretation
- Context-aware medical reasoning

#### Need
Existing systems manage data. They do not understand it.

There is currently no privacy-preserving, medically grounded, agentic AI layer that orchestrates the full lifecycle of a clinical trial.

**ACTION** combines MedGemma-powered clinical reasoning with federated intelligence to accelerate trials without sacrificing safety or governance.

#### Who Benefits and How

| User | Current Pain | ACTION's Improved Journey |
|---|---|---|
| **Research Coordinator** | 15–20 hrs/week manually screening | Natural-language eligibility → automated, explainable FHIR-based screening in minutes |
| **Biostatistician** | Delayed cohort formation & analytics | Conversational queries trigger automated survival analysis with cited outputs |
| **Principal Investigator** | Delayed safety signal detection | Real-time federated adverse event monitoring |
| **Sponsor** | Fragmented tools across trial stages | Unified AI orchestration from screening to monitoring |  

<br>

---

## Overall solution
#### MedGemma as the Clinical Reasoning Engine

**ACTION** is built around a simple principle:

> Use MedGemma only where medical reasoning is indispensable.

Across screening, cohort analytics and monitoring, MedGemma functions as a clinical decision engine embedded inside a privacy-preserving workflow.

#### Stage 1 - Intelligent Patient Screening
##### The Problem
Eligibility criteria are written in nuanced medical language and applied to messy, real-world EHR data. Rule-based systems break under ambiguity.

##### Our Approach
MedGemma reads full FHIR bundles conditions, labs, medications, notes and evaluates eligibility holistically, like a clinician.

We implement a **self-correcting two-agent system**:
- A Screener Agent makes the initial decision.
- An Auditor Agent independently re-evaluates it.
- Conflicts trigger reflection and human review.

##### Why It Matters

- Screening shifts from weeks of manual chart review to minutes with confidence scores and audit trails. 
- All processing runs locally at each site. Only aggregate counts leave the hospital.

#### Stage 2 - Accelerated Cohort Formation with AI-Guided Synthetic Control Arms
##### The Problem
In most clinical trials, eligible patients identified in Stage 1 are divided into two groups:
- Treatment arm : Receives the experimental therapy
- Control arm : Receives standard-of-care or placebo for comparison

Control arm recruitment is often the slowest part of cohort formation. In case of life-threatening diseases, where patient numbers are limited, assigning patients to placebo arms presents **significant ethical challenges**.

Meanwhile, prior trials and historical patient data often already describe how similar control arm populations behave under standard care yet this knowledge remains underutilized.

##### Our Approach
Accelerating cohort formation by enabling an AI-guided synthetic control arm built from matched historical patient data.

Instead of recruiting every control patient from scratch, the system:
- Identifies clinically comparable historical patients
- Generates baseline cohort summaries and survival benchmarks using ReAct (Reason + Act) agent
- Enables trend analytics across treatment vs historical control populations

##### Why It Matters
By reducing control arm recruitment burden, trials can:
- Lighten the recruitment workload to accelerate timelines
- Lower costs
- Limit patient exposure to placebo, enhancing ethical acceptability in serious conditions such as cancer.

We are not weakening scientific rigor.
We are accelerating trials by responsibly leveraging existing clinical knowledge.

#### Stage 3 - Federated Safety Monitoring
##### The Problem
Clinical trials generate rich, continuous data across multiple sites from lab results and imaging reports to clinician notes and adverse event narratives. Transforming this distributed information into timely, actionable insight requires intelligent, real-time synthesis to understand drug efficacy, safety trends, and patient outcomes.

##### Our Approach
Enables real-time, federated monitoring across all participating sites.

At each site, MedGemma:
- Extracts adverse events
- Identifies treatment response
- Tracks lab abnormalities and clinical trends
- Detects emerging safety patterns in notes

Structured summaries are aggregated locally and merged via federated infrastructure without transmitting patient-level data.

Investigators can ask:

> “What is the Grade 3+ adverse event rate this month?” <br>
> “Is progression-free survival trending differently at any site?”

And receive near real-time, clinically grounded answers.

##### Why It Matters
Monitoring shifts from reactive to proactive.
- Safety risks are identified earlier
- Drug efficacy signals emerge faster
- Cross-site inconsistencies are surfaced immediately
- Privacy remains intact

---

## Technical details 
#### Architecture Overview

**ACTION Platform** is deployed as a containerized, production-ready stack:
- FastAPI backend for async APIs
- MedGemma for on-premise inference
- Flower for federated coordination across sites
- Cox PH survival engine (lifelines) for deterministic analytics
- FHIR R4 bundles as interoperable clinical input

The system cleanly separates:
- Clinical Reasoning (MedGemma)
- Statistical Computation (Deterministic engines)
- Federated Aggregation (Flower framework)

This ensures medical reasoning is AI-driven, while all quantitative outputs remain auditable and reproducible.

#### Model Configuration & HAI-DEF Usage

We use pre-trained MedGemma **without fine-tuning**.

MedGemma is used exclusively for:
- Eligibility reasoning
- Clinical note extraction
- Workflow recommendations

Key safeguards:
- Low-temperature inference (0.1–0.3)
- Deterministic JSON outputs
- Self-correcting screening (screen → audit → reflect)
- Human-review flags for low-confidence decisions

The model augments clinicians, it does not replace them.

#### Privacy & Deployment
- FHIR patient data never leaves local sites.
- MedGemma runs fully on-premise
- Only aggregate metrics are federated.

#### Performance & Validation
- Screening reduces manual review workload by ~70–85% via confidence stratification.
- Synthetic control arms generate immediate baseline and survival benchmarks.
- Federated monitoring aggregates cross-site signals in seconds without sharing raw data.

#### Practical Feasibility
Designed for real-world clinical use:
- Human-in-the-loop safeguards
- Structured outputs for auditability
- Modular tool registry for extensibility
- Clear separation between AI reasoning and statistical engines

---

## Impact Potential

**1. Screening Acceleration**

- Consider an oncology trial that screens 200–500 patients, the enrollment would be around 50 reflecting an 80–85% screening failure rate *(Fogel, 2018; Tufts CSDD)*. 
- Assuming 15–20 minutes per manual eligibility review, screening 300 patients would require approximately 75–100 coordinator hours.
- **ACTION** automates high-confidence decisions, routing only ~25% of cases to manual review — reducing coordinator burden to ~25 hours, a 75% reduction. Assuming a fully-loaded CRC cost of ~$40 – 60 per hour (based on *BLS* wage data and institutional overhead adjustments), this saves **$1,000 – $1,500** per trial in direct labor, while accelerating time-to-first-patient-enrolled.

**2. Synthetic Control Arm Impact**

- In a 100-patient trial with approx. 40% control arm allocation, replacing even a portion (say 25%) of the control arm with validated historical data, consistent with FDA guidance on externally controlled trials *(FDA, 2023)* removes ~10 patients from active recruitment. 
- Across 5 sites enrolling at ~2 patients per site per month *(Getz et al., 2017)*, this shortens enrollment by approximately 4 – 6 weeks. Published estimates suggest per-patient costs in late-phase oncology trials are in the tens of thousands of dollars *(Sertkaya et al., 2016)*; removing ~10 control-arm recruits therefore represents approximately **$200,000 – $500,000** in avoided direct recruitment costs, while **reducing patient exposure to placebo or suboptimal care**.

**3. Monitoring Impact**

- Large Phase II–III oncology programs are widely reported to require substantial operational expenditure, often reaching **hundreds of millions of dollars** over their lifecycle *(Wouters et al., 2020; industry analyses)*
- If federated real-time monitoring enables go/no-go decisions 5–10% earlier in a 12-month monitoring phase, the trial concludes 3–5 weeks sooner.
- Earlier futility or superiority detection also prevents continued patient exposure to ineffective regimens, a direct ethical benefit in oncology.

**Cumulative Estimate** <br>
While exact savings depend on trial size, indication, and geographic footprint, conservative modeling suggests that:
- Screening automation reduces manual labor burden by **~70–75%**.
- Partial synthetic control substitution can reduce active recruitment needs.
- Earlier monitoring-driven decisions may shorten trial duration by **5–10%**.

Even modest percentage improvements across these domains compound meaningfully in multi-site oncology trials where operational expenditure is substantial.

> References: Fogel (2018); Tufts CSDD; Getz et al. (2017); Sertkaya et al. (2016); BLS (2023); FDA (2023); Wouters et al. (2020).