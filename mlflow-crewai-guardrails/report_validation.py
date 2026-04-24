"""
Heuristic numeric validation: flags numbers in the synthesis that don't
appear in the research sources.

``validate_report_numbers_against_sources`` is ``@mlflow.trace``-decorated;
``ValidateReportNumbersTool`` wraps it for the synthesist agent.

Span attributes recorded:
  validation.duration_ms, validation.final_report_chars,
  validation.source_union_chars, validation.report_numbers_count,
  validation.source_numeric_tokens_count, validation.source_distinct_numeric_tokens,
  validation.suspicious_count, validation.suspicious_unique_count
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterable

import mlflow
from crewai.tools import BaseTool

import crew_state

_NUM_PATTERN = re.compile(
    r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?",
    re.UNICODE,
)


def _normalize_token(tok: str) -> str:
    t = tok.strip().replace(",", "")
    if t.endswith("%"):
        t = t[:-1].strip()
    try:
        return f"{float(t):g}" if ("." in t or "e" in t.lower()) else str(int(float(t)))
    except ValueError:
        return t


def extract_numeric_tokens(text: str) -> list[str]:
    """Return all numeric tokens found in *text*."""
    if not text:
        return []
    return [m.group(0).strip() for m in _NUM_PATTERN.finditer(text)]


def _normalized_set(tokens: Iterable[str]) -> set[str]:
    out: set[str] = set()
    for t in tokens:
        out.add(t)
        out.add(_normalize_token(t))
    return {x for x in out if x}


def _set_span_attrs(attrs: dict[str, object]) -> None:
    span = mlflow.get_current_active_span()
    if span is not None:
        span.set_attributes(attrs)


@mlflow.trace(
    name="numeric_report_validation",
    attributes={"service": "heuristic numeric cross-check (report vs sources)"},
)
def validate_report_numbers_against_sources(
    final_report: str,
    *source_texts: str,
) -> str:
    """Compare numeric tokens in *final_report* to *source_texts*.

    Returns a short markdown audit report.
    """
    t0 = time.perf_counter()
    report_nums: list[str] = []
    source_tokens: list[str] = []
    suspicious: list[str] = []
    source_union = ""

    try:
        report_nums = extract_numeric_tokens(final_report)
        for s in source_texts:
            if s:
                source_union += "\n" + s

        source_tokens = extract_numeric_tokens(source_union)
        src_set = _normalized_set(source_tokens)
        compact = source_union.replace(" ", "").replace("\n", "")

        for n in report_nums:
            if _normalize_token(n) in src_set or n in src_set:
                continue
            if n.replace(" ", "") in compact:
                continue
            suspicious.append(n)

        lines = [
            "## Numeric validation (heuristic)",
            "",
            f"- Numbers in final report: **{len(report_nums)}**",
            f"- Distinct numeric tokens in sources: **{len(set(source_tokens))}**",
            "",
        ]
        if not suspicious:
            lines.append(
                "**Result:** All numeric tokens match the source text.  "
                "Still verify critical figures manually."
            )
        else:
            uniq = list(dict.fromkeys(suspicious))[:40]
            lines.append(
                "**Result:** The following appear in the report but **not** in "
                "sources (possible hallucination—review manually):"
            )
            lines.append("")
            for u in uniq:
                lines.append(f"- `{u}`")
            if len(suspicious) > 40:
                lines.append(f"\n*…and {len(suspicious) - 40} more*")

        lines.extend(["", "_Programmatic check; may false-positive on reformatted numbers._"])
        return "\n".join(lines)
    finally:
        _set_span_attrs({
            "validation.duration_ms": round((time.perf_counter() - t0) * 1000, 3),
            "validation.final_report_chars": len(final_report),
            "validation.source_union_chars": len(source_union),
            "validation.report_numbers_count": len(report_nums),
            "validation.source_numeric_tokens_count": len(source_tokens),
            "validation.source_distinct_numeric_tokens": len(set(source_tokens)),
            "validation.suspicious_count": len(suspicious),
            "validation.suspicious_unique_count": len(dict.fromkeys(suspicious)),
        })


class ValidateReportNumbersTool(BaseTool):
    """CrewAI tool wrapping ``validate_report_numbers_against_sources``."""

    name: str = "validate_report_numbers_against_sources"
    description: str = (
        "Heuristic numeric audit: compares every number in your synthesis to "
        "the research brief stored in the shared database.  Pass your complete "
        "synthesis text as final_report.  Returns a short markdown report.  "
        "Call once after drafting, then fold the result into your output."
    )

    def _run(self, final_report: str) -> str:
        research = crew_state.get_state("research_brief") or ""
        return validate_report_numbers_against_sources(final_report, research)
