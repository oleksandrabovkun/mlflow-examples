"""
CrewAI multi-agent financial-analysis crew with MLflow tracing and
AI governance guardrails.

Four agents run in sequence:
  1. Orchestrator — scopes the engagement charter.
  2. Macro Data Specialist — fetches World Bank CPI inflation (nested span).
  3. Research Analyst — writes the household-pressure research brief.
  4. Synthesist — produces the investment-style synthesis and calls
     the numeric-validation tool.

Every agent, task, tool call, and LLM completion is auto-traced via
``mlflow.crewai.autolog()`` + ``mlflow.openai.autolog()``.  After the crew
finishes, ``operational_governance.analyze_last_trace()`` inspects the
completed trace for governance signals (e.g. redundant tool-call loops).
"""

from __future__ import annotations

import os
import time
from textwrap import dedent

from crewai import Agent, Crew, Task
from crewai.crews.crew_output import CrewOutput
from crewai.llm import LLM
from crewai.llms.base_llm import BaseLLM
from dotenv import load_dotenv
import mlflow

import crew_state
import operational_governance
from guardrails import GuardedFetchWorldBankInflationTool
from pii_redaction import redact_pii
from report_validation import (
    ValidateReportNumbersTool,
    extract_numeric_tokens,
    validate_report_numbers_against_sources,
)

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def build_llm() -> BaseLLM:
    """Build a CrewAI-native OpenAI LLM from environment variables."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY is not set.  Copy .env.example → .env and add your key."
        )
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip()
    if not model:
        raise ValueError("OPENAI_MODEL must not be empty when set.")
    base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
    kwargs: dict = {"model": model, "api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url.rstrip("/")
    return LLM(**kwargs)

# ---------------------------------------------------------------------------
# Topic
# ---------------------------------------------------------------------------

TOPIC = dedent("""\
    How ordinary households are squeezed when inflation, interest rates, and
    shelter costs move in different directions: affordability of rent and
    mortgages, food and energy bills, real wages, debt payments (cards, loans),
    and emergency savings—across typical high-income and emerging economies,
    not one country's policy framework.  What drives the pressure, who is most
    exposed, and what to watch over the next 12–24 months.""")

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
# GuardedFetchWorldBankInflationTool is imported from guardrails.py.
# It wraps the base fetch tool with a call-count circuit breaker that blocks
# execution if the tool is called more than TOOL_CALL_LIMIT times per run.

# ---------------------------------------------------------------------------
# Callbacks — persist task outputs + collect metrics for the root span
# ---------------------------------------------------------------------------

_crew_metrics: dict[str, object] = {}


def _on_plan_complete(output: object) -> None:
    text = crew_state.task_output_to_text(output)
    crew_state.set_state("engagement_charter", text)
    _crew_metrics["orchestration_lead.output_chars"] = len(text)


def _on_stats_complete(output: object) -> None:
    text = crew_state.task_output_to_text(output)
    crew_state.set_state("macro_stats_snapshot", text)
    _crew_metrics["macro_data_specialist.output_chars"] = len(text)


def _on_research_complete(output: object) -> None:
    text = crew_state.task_output_to_text(output)
    crew_state.set_state("research_brief", text)
    _crew_metrics["research_analyst.output_chars"] = len(text)


def _on_synthesis_complete(output: object) -> None:
    synthesis = crew_state.task_output_to_text(output)
    crew_state.set_state("synthesis_result", synthesis)
    _crew_metrics["synthesist.output_chars"] = len(synthesis)
    research = crew_state.get_state("research_brief") or ""
    result = validate_report_numbers_against_sources(synthesis, research)
    _crew_metrics["validation.result_chars"] = len(result)

# ---------------------------------------------------------------------------
# Crew assembly
# ---------------------------------------------------------------------------

def build_crew(llm: BaseLLM, topic: str = TOPIC) -> Crew:
    """Assemble four agents and four sequential tasks.

    Parameters
    ----------
    topic:
        The research topic passed into task descriptions.  Defaults to the
        module-level ``TOPIC`` constant but can be overridden — e.g. to pass
        a PII-redacted version of user-supplied input.
    """

    orchestrator = Agent(
        role="Engagement Orchestration Lead",
        goal=(
            "Turn a high-level investment question into a crisp research "
            "mandate: scope, key metrics, risks, and the exact deliverables "
            "sub-agents must produce."
        ),
        backstory=(
            "You are a senior engagement manager at a buyside equity research "
            "desk.  You never do primary number-crunching yourself—you define "
            "what 'good' looks like so specialists can execute without ambiguity."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    macro_data = Agent(
        role="Macro Data Specialist",
        goal=(
            "Fetch one authoritative CPI inflation series from the World Bank "
            "public API and summarize it so downstream analysts can cite it."
        ),
        backstory=(
            "You only use the fetch_world_bank_inflation tool (World Bank JSON "
            "API).  You pick a 2-letter ISO country code relevant to the "
            "engagement, call the tool once, and return the raw tool output "
            "plus a short interpretation.  You do not draft the full research "
            "brief—that is another agent."
        ),
        tools=[GuardedFetchWorldBankInflationTool()],
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    researcher = Agent(
        role="Household Economics & Consumer Finance Research Analyst",
        goal=(
            "Produce a structured research brief on how macro conditions "
            "(inflation, policy rates, labor markets, housing markets) transmit "
            "to typical family budgets: shelter, essentials, debt service, and "
            "savings buffers—using cross-country patterns, not a single market's "
            "rulebook."
        ),
        backstory=(
            "You translate macro and household-finance research for "
            "non-specialists.  You name mechanisms (e.g., floating vs fixed "
            "debt, rent pass-through, real wage growth), note where data is "
            "country-specific, and separate established relationships from open "
            "debates.  The Macro Data task already fetched World Bank CPI "
            "inflation—integrate and cite that block from your task context."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    synthesist = Agent(
        role="Portfolio Financial Synthesist",
        goal=(
            "Deliver a concise investment-style outcome: thesis, bull/base/bear "
            "cases, metric watchlist, and how you would express the view."
        ),
        backstory=(
            "You are a PM-aligned strategist: you turn research into actionable "
            "views, probability-weighted narratives, and plain English risk "
            "disclosures.  Call validate_report_numbers_against_sources after "
            "drafting to attach a numeric sanity-check."
        ),
        tools=[ValidateReportNumbersTool()],
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    # -- Tasks (sequential) ------------------------------------------------

    plan_task = Task(
        description=dedent(f"""\
            Topic for the engagement:
            {topic}

            Produce:
            1) Problem statement in 3–5 bullets.
            2) Research questions (ordered) the analyst must answer.
            3) Required sections for the research brief (with purpose each).
            4) Explicit non-goals / out of scope.
            5) Success criteria for the final synthesis."""),
        expected_output=(
            "A structured engagement charter (markdown) with numbered sections "
            "exactly as listed; no preliminary research—planning only."
        ),
        agent=orchestrator,
        callback=_on_plan_complete,
    )

    stats_task = Task(
        description=dedent(f"""\
            Engagement topic (for country choice):
            {topic}

            From the orchestration charter in context, pick one major economy.
            Call fetch_world_bank_inflation exactly once with that country's
            2-letter ISO code.

            Output:
            1) The full tool response (verbatim block).
            2) One short paragraph explaining how this CPI series fits the
               household budget story."""),
        expected_output="World Bank tool output + brief interpretation (markdown).",
        agent=macro_data,
        context=[plan_task],
        callback=_on_stats_complete,
    )

    research_task = Task(
        description=dedent("""\
            Execute the research mandate from the charter.  Write the brief on
            household financial pressure in a multi-country lens.  Cover:
            - Inflation / policy-rate transmission to mortgage, rent, credit, essentials
            - Real wages vs inflation
            - Shelter stress: renters vs owners, fixed vs floating debt
            - Debt vulnerability: unsecured balances, BNPL, stress signals
            - Savings buffers and inequality of exposure
            - Regional variation (cite concepts, not one regulator's filing)

            Weave in the World Bank CPI data from context with attribution.
            Use clear headings.  Mark estimates vs facts.  No buy/sell—that
            comes later.  Output full markdown with multiple sections."""),
        expected_output=(
            "Structured markdown research brief with quantitative placeholders "
            "where firm-specific data is required."
        ),
        agent=researcher,
        context=[plan_task, stats_task],
        callback=_on_research_complete,
    )

    synthesis_task = Task(
        description=dedent("""\
            Read the charter and research brief from task context.  Write:
            - One-paragraph thesis
            - Bull / base / bear narratives (2–3 bullets each)
            - Metrics and events to monitor (table or bullets)
            - How to express the view and what would flip it
            - Plain-language risk disclosure (not personal advice)

            After drafting, call validate_report_numbers_against_sources once
            with your full synthesis as final_report.  Append the returned
            markdown under ## Numeric validation (heuristic).

            Use normal markdown—do not emit raw JSON tool calls."""),
        expected_output=(
            "Executive markdown: thesis, scenarios, metrics, expression, risks."
        ),
        agent=synthesist,
        context=[plan_task, research_task],
        callback=_on_synthesis_complete,
    )

    return Crew(
        agents=[orchestrator, macro_data, researcher, synthesist],
        tasks=[plan_task, stats_task, research_task, synthesis_task],
        verbose=True,
    )

# ---------------------------------------------------------------------------
# Root traced wrapper — sets summary attributes on the top-level span
# ---------------------------------------------------------------------------

@mlflow.trace(name="Crew.kickoff", span_type="CHAIN")
def run_crew_with_metrics(crew: Crew, topic: str) -> CrewOutput:
    """Run ``crew.kickoff()`` inside a traced span that collects summary metrics.

    PII redaction runs here so it becomes a child span of the root trace
    rather than a separate top-level trace.
    """
    _crew_metrics.clear()
    t0 = time.perf_counter()

    redact_pii(topic)

    result = crew.kickoff()

    duration_s = round(time.perf_counter() - t0, 3)
    span = mlflow.get_current_active_span()
    if span is not None:
        charter = crew_state.get_state("engagement_charter") or ""
        stats = crew_state.get_state("macro_stats_snapshot") or ""
        research = crew_state.get_state("research_brief") or ""
        synthesis = crew_state.get_state("synthesis_result") or ""

        span.set_attributes({
            "crew.total_duration_s": duration_s,
            "crew.task_count": len(crew.tasks),
            "crew.agent_count": len(crew.agents),
            "orchestration_lead.output_chars": len(charter),
            "macro_data_specialist.output_chars": len(stats),
            "research_analyst.output_chars": len(research),
            "synthesist.output_chars": len(synthesis),
            "validation.report_number_count": len(extract_numeric_tokens(synthesis)),
            "validation.source_number_count": len(extract_numeric_tokens(research)),
            **_crew_metrics,
        })

    return result

# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def _task_raw(crew_output: CrewOutput, index: int) -> str:
    tasks = crew_output.tasks_output
    if index < len(tasks) and tasks[index].raw:
        return str(tasks[index].raw).strip()
    return ""


def print_run_results(crew_output: CrewOutput | object) -> None:
    """Pretty-print crew deliverables to stdout."""
    w = 78
    bar = "=" * w
    if not isinstance(crew_output, CrewOutput):
        print(f"\n{bar}\n{str(crew_output).strip()}\n{bar}\n")
        return

    print(f"\n{bar}")
    print(" CREW OUTPUT ".center(w))
    print(bar)

    labels = [
        "1. Engagement charter",
        "2. Macro stats (World Bank)",
        "3. Research brief",
        "4. Synthesis",
    ]
    for i, label in enumerate(labels):
        body = _task_raw(crew_output, i)
        print(f"\n--- {label} ---\n")
        print(body or "(no text captured — see crew logs above)")
    print()

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv()
    crew_state.init_db()

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    experiment = os.environ.get(
        "MLFLOW_EXPERIMENT_NAME", "crewai-household-financial-pressure"
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    mlflow.crewai.autolog()
    mlflow.openai.autolog()

    crew = build_crew(build_llm(), topic=TOPIC)
    result = run_crew_with_metrics(crew, topic=TOPIC)
    print_run_results(result)

    # --- Operational Governance -------------------------------------------
    # Analyse the completed trace for health signals.  Currently checks for
    # redundant tool-call loops; the report is printed to stdout and can be
    # extended with additional operational, compliance, cost, and quality checks.
    operational_governance.analyze_last_trace()


if __name__ == "__main__":
    main()
