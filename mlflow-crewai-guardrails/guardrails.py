"""
Active Guardrails — real-time enforcement for CrewAI tool calls.

Governance category: **Operational + Compliance**
Answers: Can we prevent unsafe or runaway agent behavior before it happens?

Each guarded tool wraps the underlying implementation with a pre-execution
check.  When a guardrail fires, it:
  1. Sets a ``guardrail.blocked = True`` attribute on the active MLflow span,
     making the enforcement event visible in the trace DAG.
  2. Raises an exception that CrewAI surfaces as a task failure, stopping
     the problematic execution path.

This approach has two advantages over intercepting spans mid-flight:
  - The check is synchronous and happens before any external call is made.
  - The guardrail span is a first-class citizen of the MLflow trace, so
    enforcement events are queryable alongside operational metrics.
"""

from __future__ import annotations

import mlflow
from crewai.tools import BaseTool

from web_stats import fetch_world_bank_inflation_summary

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Maximum number of times any single tool instance may be called per crew run.
#: Exceeding this limit is treated as a retry loop and the call is blocked.
TOOL_CALL_LIMIT: int = 0


# ---------------------------------------------------------------------------
# Guarded tool: World Bank inflation fetch
# ---------------------------------------------------------------------------

class GuardedFetchWorldBankInflationTool(BaseTool):
    """World Bank CPI fetch tool with a call-count circuit breaker.

    Inherits the description from the base implementation so agent prompts
    remain unchanged.  Adds a guardrail that blocks execution if the tool
    is called more than ``TOOL_CALL_LIMIT`` times in a single crew run,
    which indicates a retry loop rather than intentional multi-country
    research.
    """

    name: str = "fetch_world_bank_inflation"
    description: str = (
        "Loads annual CPI inflation (% YoY) from the World Bank public JSON API.  "
        "Pass a 2-letter ISO country code (e.g. US, GB, DE, JP).  "
        "Use once to anchor a quantitative thread; cite World Bank in the brief."
    )

    # Pydantic fields for call tracking (private, not exposed to the LLM)
    _call_count: int = 0

    def _run(self, country_code: str = "US") -> str:
        self._call_count += 1

        span = mlflow.get_current_active_span()
        if span:
            span.set_attribute("guardrail.call_count", self._call_count)
            span.set_attribute("guardrail.call_limit", TOOL_CALL_LIMIT)

        if self._call_count > TOOL_CALL_LIMIT:
            if span:
                span.set_attribute("guardrail.blocked", True)
                span.set_attribute(
                    "guardrail.reason",
                    f"call_count_exceeded (count={self._call_count}, limit={TOOL_CALL_LIMIT})",
                )
            raise RuntimeError(
                f"Guardrail blocked: fetch_world_bank_inflation called "
                f"{self._call_count} times in this run (limit: {TOOL_CALL_LIMIT}). "
                "Possible retry loop — stopping execution to prevent runaway cost."
            )

        return fetch_world_bank_inflation_summary(country_code.strip())


# ---------------------------------------------------------------------------
# Cost / authorization guardrail (template for financial tools)
# ---------------------------------------------------------------------------

def require_mfa_for_large_transfers(
    amount: float,
    account_id: str,
    mfa_verified: bool = False,
) -> None:
    """Pre-execution check for high-value financial operations.

    Raises ``PermissionError`` if a transfer exceeds $10,000 without MFA
    confirmation.  Call this at the top of any tool that executes financial
    transactions before touching external systems.

    Parameters
    ----------
    amount:
        The transaction amount in USD.
    account_id:
        The destination account identifier (logged to the span for audit).
    mfa_verified:
        Whether the requesting user has completed MFA.  Must be ``True``
        for amounts above the threshold.
    """
    span = mlflow.get_current_active_span()
    if span:
        span.set_attribute("guardrail.amount", amount)
        span.set_attribute("guardrail.account_id", account_id)
        span.set_attribute("guardrail.mfa_verified", mfa_verified)

    if amount > 10_000 and not mfa_verified:
        if span:
            span.set_attribute("guardrail.blocked", True)
            span.set_attribute("guardrail.reason", "MFA_REQUIRED")
        raise PermissionError(
            f"Guardrail blocked: MFA verification required for transfers "
            f"over $10,000 (requested: ${amount:,.2f})."
        )
