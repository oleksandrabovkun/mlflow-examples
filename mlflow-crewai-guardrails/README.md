# MLflow + CrewAI Governance Guardrails Demo

A multi-agent [CrewAI](https://www.crewai.com/) crew that analyzes **household financial pressure** (inflation, rates, shelter, wages, debt) across countries, with every agent, tool call, and LLM completion traced through **[MLflow](https://mlflow.org/)** — and a set of **AI governance guardrails** that span detection, compliance, cost attribution, and active enforcement.

Companion code for the blog post series:
- Part 1: Multi-Agent Observability in Production with MLflow ([`mlflow-crewai-observability`](../mlflow-crewai-observability))
- Part 2: Multi-Agent Governance in Production with MLflow ← this repo

## Governance dimensions

| Dimension | Status | What it answers |
|-----------|--------|-----------------|
| **Operational** | ✅ implemented | Is the agent system healthy and behaving efficiently? |
| **Compliance & Audit** | ✅ implemented | Does the system mask PII and protect credentials? |
| **Cost** | ✅ implemented | Is the agent's trajectory token- and compute-efficient? |
| **Quality & Value** | 🔜 planned | Does the system deliver value and meet user expectations? |

---

## Operational governance — redundant loop detection

After each crew run, `operational_governance.analyze_last_trace()` fetches the completed MLflow trace and counts how many times each tool was called. If any tool exceeds the configured threshold (default **5 calls per trace**), a governance warning is printed:

```
Operational Warning: Agent entered a redundant loop.
Tool 'fetch_world_bank_inflation' was called 7 times in a single trace (threshold: 5).
```

The `LoopReport` object returned can be consumed programmatically to drive further actions (abort the run, fire an alert, write a metric to an observability platform).

Key configuration in `operational_governance.py`:

| Constant | Default | Purpose |
|----------|---------|---------|
| `MONITORED_TOOLS` | `{"fetch_world_bank_inflation", "validate_report_numbers_against_sources", "search_tool"}` | Tool span names to track |
| `REDUNDANT_LOOP_THRESHOLD` | `5` | Alert when a tool is called more than this many times |

`operational_governance.py` also exposes `calculate_trajectory_cost(trace)`, which aggregates token usage per span across the full trace — useful for identifying which agent steps are most expensive.

---

## Compliance & audit — PII redaction

`pii_redaction.redact_pii()` strips common PII patterns (email, phone, credit card, SSN) from any user-supplied text before it enters the agent pipeline. The function is wrapped in `@mlflow.trace`, so the redaction itself appears as a span in the MLflow trace DAG alongside agent spans:

- You can verify that masking happened before the LLM was called.
- The span records which pattern types were matched and how many substitutions were made (`pii.email_count`, `pii.total_redactions`, etc.).

In `financial_crew.py`, the research topic is redacted before being passed to `build_crew()`:

```python
safe_topic = redact_pii(TOPIC)
crew = build_crew(build_llm(), topic=safe_topic)
```

---

## Active guardrails — circuit breakers

`guardrails.py` wraps the World Bank fetch tool with a call-count circuit breaker that **blocks execution** if the tool is called more than `TOOL_CALL_LIMIT` (3) times in a single run.

When the guardrail fires:
1. `guardrail.blocked = True` is set on the active MLflow span — enforcement is visible in the trace.
2. A `RuntimeError` is raised, which CrewAI surfaces as a task failure.

The check is synchronous and happens before any external call is made.

`guardrails.py` also includes `require_mfa_for_large_transfers()` — a template for financial authorization guardrails that block high-value operations without MFA confirmation.

---

## What the crew does

Four AI agents run in sequence, each with a distinct role:

| # | Agent | Responsibility | MLflow visibility |
|---|-------|----------------|-------------------|
| 1 | **Orchestration Lead** | Scopes the engagement charter | Agent span |
| 2 | **Macro Data Specialist** | Fetches World Bank CPI inflation | Agent → Tool → `world_bank_stats_api` nested span |
| 3 | **Research Analyst** | Writes a structured research brief | Agent span |
| 4 | **Portfolio Synthesist** | Produces the investment-style synthesis | Agent → Tool → `numeric_report_validation` nested span |

```text
pii_redaction                          ← @mlflow.trace (masks topic before crew assembly)
Crew.kickoff (root span — summary attributes)
├── Task 1 → Orchestration Lead
├── Task 2 → Macro Data Specialist
│   └── fetch_world_bank_inflation     ← GuardedFetchWorldBankInflationTool (circuit breaker)
│       └── world_bank_stats_api       ← @mlflow.trace (HTTP metrics)
├── Task 3 → Research Analyst
└── Task 4 → Portfolio Synthesist
    └── numeric_report_validation      ← @mlflow.trace (validation metrics)

Post-run: operational_governance.analyze_last_trace()
└── Reads the completed trace and checks for redundant tool-call loops
```

---

## Prerequisites

- Python **3.10+**
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
- An **OpenAI API key** ([get one here](https://platform.openai.com/api-keys))

## Quickstart

```bash
# 1. Clone and enter the repo
git clone <repo-url> && cd mlflow-crewai-guardrails

# 2. Install dependencies
uv sync

# 3. Configure your API key
cp .env.example .env
# Edit .env and set OPENAI_API_KEY

# 4. Run the crew (governance report prints automatically after the run)
uv run python financial_crew.py

# 5. Start the MLflow UI (in a separate terminal)
uv run mlflow server --backend-store-uri sqlite:///mlflow.db
# Open http://127.0.0.1:5000
```

## Configuration

All settings go in `.env` (see `.env.example`):

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `OPENAI_API_KEY` | Yes | — | OpenAI Chat Completions |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | Model to use |
| `OPENAI_BASE_URL` | No | — | Azure OpenAI / proxy (also accepted as `OPENAI_API_BASE`) |
| `MLFLOW_TRACKING_URI` | No | `sqlite:///mlflow.db` | MLflow backend store |
| `MLFLOW_EXPERIMENT_NAME` | No | `crewai-household-financial-pressure` | Experiment name |
| `CREW_STATE_DB` | No | `crew_state.db` | SQLite shared state path |

## Project layout

```
├── financial_crew.py          # Crew definition, agents, tasks, MLflow setup, main()
├── guardrails.py              # Active guardrails: circuit breakers for tool calls and financial ops
├── pii_redaction.py           # Compliance: PII redaction with MLflow trace audit trail
├── operational_governance.py  # Operational checks: loop detection, cost attribution
├── web_stats.py               # World Bank JSON client + @mlflow.trace
├── report_validation.py       # Numeric validation tool + @mlflow.trace
├── crew_state.py              # SQLite shared state (task callbacks)
├── pyproject.toml             # Dependencies (uv)
├── .env.example               # Environment template
└── .gitignore
```

## Notes

- Output is **illustrative** — not personal financial advice.
- MLflow's CrewAI integration traces synchronous `kickoff()` only; see [MLflow CrewAI docs](https://mlflow.org/docs/latest/tracing/integrations/crewai).
- The World Bank API is public and free; no API key needed.
