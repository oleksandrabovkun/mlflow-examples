"""
Operational Governance — post-run analysis of MLflow traces.

Governance category: **Operational**
Answers: Is the agent system healthy and behaving efficiently?

Checks implemented
------------------
1. Redundant loop detection
   Flags when the same tool is called an excessive number of times within a
   single trace, which indicates the agent entered a retry/search loop without
   converging.  The telemetry surfaced here informs future hard guardrails
   (e.g. capping tool invocations at 3 per task).

Future checks (placeholders for other governance dimensions):
- Compliance & Audit: verify that agent outputs contain mandatory disclosures.
- Cost: compute estimated token spend and flag traces that exceed a budget.
- Quality & Value: score synthesis coherence against user-supplied criteria.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import mlflow

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Tools whose call count is tracked for loop detection.
MONITORED_TOOLS: frozenset[str] = frozenset({
    "fetch_world_bank_inflation",
    "validate_report_numbers_against_sources",
    "search_tool",          # placeholder for future tools
})

#: Alert threshold: more than this many calls to one tool per trace is flagged.
REDUNDANT_LOOP_THRESHOLD: int = 5


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class LoopReport:
    """Summary of redundant loop detection for a single trace."""

    trace_id: str
    tool_call_counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def has_warnings(self) -> bool:
        return bool(self.warnings)


# ---------------------------------------------------------------------------
# Operational checks
# ---------------------------------------------------------------------------

def detect_redundant_loops(
    trace_id: str,
    *,
    threshold: int = REDUNDANT_LOOP_THRESHOLD,
    monitored_tools: frozenset[str] = MONITORED_TOOLS,
) -> LoopReport:
    """Analyse *trace_id* for tool calls that exceed *threshold*.

    Parameters
    ----------
    trace_id:
        The MLflow trace ID returned by ``mlflow.get_last_active_trace_id()``
        or stored on a ``RunData`` object.
    threshold:
        Maximum acceptable number of calls to any single tool.  Calls above
        this count are flagged as a potential redundant loop.
    monitored_tools:
        Set of tool span names to track.  Spans whose ``name`` field matches
        any entry are counted.  Pass an empty set to count *all* tool spans.

    Returns
    -------
    LoopReport
        Contains per-tool call counts and any governance warnings.
    """
    trace = mlflow.get_trace(trace_id)
    report = LoopReport(trace_id=trace_id)

    # Count every span whose name matches a monitored tool (or all tools if
    # monitored_tools is empty, which means "count everything").
    all_spans = trace.data.spans
    tool_spans = [
        s for s in all_spans
        if (not monitored_tools) or (s.name in monitored_tools)
    ]

    counts: Counter[str] = Counter(s.name for s in tool_spans)
    report.tool_call_counts = dict(counts)

    for tool_name, call_count in counts.items():
        if call_count > threshold:
            msg = (
                f"Operational Warning: Agent entered a redundant loop. "
                f"Tool '{tool_name}' was called {call_count} times in a single trace "
                f"(threshold: {threshold}). "
                f"This telemetry informs a future guardrail to cap tool calls at {threshold}."
            )
            report.warnings.append(msg)

    return report


def print_loop_report(report: LoopReport) -> None:
    """Print a human-readable operational governance report to stdout."""
    w = 72
    bar = "=" * w
    print(f"\n{bar}")
    print(" OPERATIONAL GOVERNANCE — Redundant Loop Detection ".center(w))
    print(bar)
    print(f"\nTrace ID : {report.trace_id}")

    if report.tool_call_counts:
        print("\nTool call counts:")
        for tool, count in sorted(report.tool_call_counts.items()):
            flag = " ⚠" if count > REDUNDANT_LOOP_THRESHOLD else ""
            print(f"  {tool}: {count}{flag}")
    else:
        print("\nNo monitored tool calls found in this trace.")

    if report.has_warnings:
        print("\nWarnings:")
        for w_msg in report.warnings:
            print(f"  • {w_msg}")
    else:
        print("\nResult: No redundant loops detected. Agent trajectory looks healthy.")

    print(f"\n{bar}\n")


# ---------------------------------------------------------------------------
# Convenience: analyse the most recent active trace
# ---------------------------------------------------------------------------

def calculate_trajectory_cost(trace) -> dict[str, int]:
    """Aggregate token usage per span name across the full crew trace.

    MLflow autolog records token counts under OpenTelemetry GenAI semantic
    conventions (``gen_ai.usage.total_tokens``) with a fallback to the older
    ``llm.usage.total_tokens`` key.  Returns a dict mapping span name to
    total tokens — sort descending to find the most expensive steps.

    Parameters
    ----------
    trace:
        An MLflow ``Trace`` object, e.g. from ``mlflow.get_trace(mlflow.get_last_active_trace_id())``.

    Returns
    -------
    dict[str, int]
        Mapping of span name → total token count.
    """
    token_counts: dict[str, int] = {}
    for span in trace.data.spans:
        attrs = span.attributes or {}
        total = (
            attrs.get("gen_ai.usage.total_tokens")
            or attrs.get("llm.usage.total_tokens")
            or 0
        )
        if total:
            token_counts[span.name] = token_counts.get(span.name, 0) + int(total)
    return token_counts


def analyze_last_trace(
    *,
    threshold: int = REDUNDANT_LOOP_THRESHOLD,
    monitored_tools: frozenset[str] = MONITORED_TOOLS,
    verbose: bool = True,
) -> LoopReport | None:
    """Retrieve the last active MLflow trace and run all operational checks.

    Returns ``None`` if no active trace is found.
    """
    trace_id = mlflow.get_last_active_trace_id()
    if trace_id is None:
        print("Operational Governance: no active trace found — skipping analysis.")
        return None
    report = detect_redundant_loops(
        trace_id,
        threshold=threshold,
        monitored_tools=monitored_tools,
    )

    if verbose:
        print_loop_report(report)

    return report
