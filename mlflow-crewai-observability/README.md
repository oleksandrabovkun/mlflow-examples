# MLflow + CrewAI Observability Demo

A multi-agent [CrewAI](https://www.crewai.com/) crew that analyzes **household financial pressure** (inflation, rates, shelter, wages, debt) across countries, with every agent, tool call, and LLM completion traced through **[MLflow](https://mlflow.org/)**.

## What it does

Four AI agents run in sequence, each with a distinct role:

| # | Agent | Responsibility | MLflow visibility |
|---|-------|----------------|-------------------|
| 1 | **Orchestration Lead** | Scopes the engagement charter | Agent span |
| 2 | **Macro Data Specialist** | Fetches World Bank CPI inflation | Agent → Tool → `world_bank_stats_api` nested span |
| 3 | **Research Analyst** | Writes a structured research brief | Agent span |
| 4 | **Portfolio Synthesist** | Produces the investment-style synthesis | Agent → Tool → `numeric_report_validation` nested span |

```text
Crew.kickoff (root span — summary attributes here)
├── Task 1 → Orchestration Lead
├── Task 2 → Macro Data Specialist
│   └── world_bank_stats_api          ← @mlflow.trace (HTTP metrics)
├── Task 3 → Research Analyst
└── Task 4 → Portfolio Synthesist
    └── numeric_report_validation     ← @mlflow.trace (validation metrics)
```

## Observability features

- **`mlflow.crewai.autolog()`** — traces crew, task, and agent execution.
- **`mlflow.openai.autolog()`** — traces every OpenAI LLM call with token counts (`prompt_tokens`, `completion_tokens`, `total_tokens`).
- **`@mlflow.trace`** on `web_stats.fetch_world_bank_inflation_summary` — adds span attributes: `stats_api.duration_ms`, `stats_api.http_status`, `stats_api.observation_rows_in_response`, etc.
- **`@mlflow.trace`** on `report_validation.validate_report_numbers_against_sources` — adds: `validation.suspicious_count`, `validation.report_numbers_count`, etc.
- **Root span attributes** — `crew.total_duration_s`, `crew.agent_count`, per-agent `*.output_chars`, validation summary.

## Prerequisites

- Python **3.10+**
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
- An **OpenAI API key** ([get one here](https://platform.openai.com/api-keys))

## Quickstart

```bash
# 1. Clone and enter the repo
git clone <repo-url> && cd mlflow-crewai-observability

# 2. Install dependencies
uv sync

# 3. Configure your API key
cp .env.example .env
# Edit .env and set OPENAI_API_KEY

# 4. Run the crew
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
| `OPENAI_BASE_URL` | No | — | Azure OpenAI / proxy |
| `MLFLOW_TRACKING_URI` | No | `sqlite:///mlflow.db` | MLflow backend store |
| `MLFLOW_EXPERIMENT_NAME` | No | `crewai-household-financial-pressure` | Experiment name |
| `CREW_STATE_DB` | No | `crew_state.db` | SQLite shared state path |

## Project layout

```
├── financial_crew.py      # Crew definition, agents, tasks, MLflow setup, main()
├── web_stats.py            # World Bank JSON client + @mlflow.trace
├── report_validation.py    # Numeric validation tool + @mlflow.trace
├── crew_state.py           # SQLite shared state (task callbacks)
├── pyproject.toml          # Dependencies (uv)
├── .env.example            # Environment template
└── .gitignore
```

## How it works

1. **`main()`** loads `.env`, initializes SQLite state, configures MLflow tracking, and enables `crewai` + `openai` autologging.
2. **`run_crew_with_metrics()`** wraps `crew.kickoff()` in a `@mlflow.trace` span. After all four tasks complete, it sets summary attributes (durations, output sizes, validation counts) on the root span.
3. **Task callbacks** persist each task's output to SQLite (`crew_state.db`) so the validation tool can cross-check the synthesis against the research brief.
4. **`validate_report_numbers_against_sources`** runs both as a tool (called by the synthesist agent) and as a callback guarantee (called in `_on_synthesis_complete`), ensuring the validation span always appears in the trace.

## Notes

- Output is **illustrative** — not personal financial advice.
- MLflow's CrewAI integration traces synchronous `kickoff()` only; see [MLflow CrewAI docs](https://mlflow.org/docs/latest/tracing/integrations/crewai).
- The World Bank API is public and free; no API key needed.
