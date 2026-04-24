"""
PII Redaction — compliance guardrail for agent inputs.

Governance category: **Compliance & Audit**
Answers: Is sensitive data being masked before it reaches an LLM agent?

Wraps redaction in an ``@mlflow.trace`` span so that every masking event is
visible in the trace DAG alongside the agent spans that consume the output.
This gives you a verifiable audit trail: you can confirm that redaction
happened, inspect which patterns were matched, and verify that the redacted
text is what the agent actually received.
"""

from __future__ import annotations

import re

import mlflow

# ---------------------------------------------------------------------------
# PII pattern definitions
# ---------------------------------------------------------------------------

#: Mapping of label → compiled regex for common PII types.
#: Extend this dict to add new entity types without changing the redaction logic.
PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "EMAIL": re.compile(
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    ),
    "PHONE": re.compile(
        r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"
    ),
    "CREDIT_CARD": re.compile(
        r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"
    ),
    "SSN": re.compile(
        r"\b\d{3}-\d{2}-\d{4}\b"
    ),
}


# ---------------------------------------------------------------------------
# Redaction function
# ---------------------------------------------------------------------------

@mlflow.trace(name="pii_redaction", span_type="TOOL")
def redact_pii(text: str) -> str:
    """Redact common PII patterns from *text* before it reaches an LLM agent.

    Each matched entity is replaced with a ``[LABEL_REDACTED]`` placeholder.
    The function is a no-op on empty input.

    Parameters
    ----------
    text:
        Raw input text that may contain sensitive data.

    Returns
    -------
    str
        Text with PII patterns replaced by labelled placeholders.

    Examples
    --------
    >>> redact_pii("Contact jane@example.com or call 555-867-5309.")
    'Contact [EMAIL_REDACTED] or call [PHONE_REDACTED].'
    """
    if not text:
        return text

    redacted = text
    matches_found: dict[str, int] = {}

    for label, pattern in PII_PATTERNS.items():
        new_text, count = pattern.subn(f"[{label}_REDACTED]", redacted)
        if count:
            matches_found[label] = count
            redacted = new_text

    # Surface match counts as span attributes so they're queryable in MLflow UI
    span = mlflow.get_current_active_span()
    if span and matches_found:
        span.set_attributes({
            f"pii.{label.lower()}_count": count
            for label, count in matches_found.items()
        })
        span.set_attribute("pii.total_redactions", sum(matches_found.values()))

    return redacted
