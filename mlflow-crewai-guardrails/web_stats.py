"""
World Bank public JSON API with ``@mlflow.trace`` instrumentation.

Called from the Macro Data agent's tool.  The traced span nests under the
active CrewAI trace so the MLflow UI shows an explicit hop to an external
stats service.

Span attributes recorded:
  stats_api.country_code, stats_api.request_url, stats_api.indicator_id,
  stats_api.duration_ms, stats_api.http_duration_ms, stats_api.http_status,
  stats_api.observation_rows_in_response, stats_api.points_in_summary,
  stats_api.output_char_count, stats_api.output_line_count
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request

import mlflow

_INDICATOR = "FP.CPI.TOTL.ZG"          # CPI inflation (annual %)
_API = "https://api.worldbank.org/v2/country"


def _http_get_json(url: str, timeout: float = 60.0) -> tuple[object, int]:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "mlflow-crewai-demo/1.0"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        raw = resp.read().decode("utf-8")
        return json.loads(raw), int(getattr(resp, "status", 200))


def _set_span_attrs(attrs: dict[str, object]) -> None:
    span = mlflow.get_current_active_span()
    if span is not None:
        span.set_attributes(attrs)


@mlflow.trace(
    name="world_bank_stats_api",
    attributes={"service": "World Bank Data API v2 (JSON)"},
)
def fetch_world_bank_inflation_summary(country_code: str = "US") -> str:
    """Fetch a short CPI inflation series from the World Bank REST API."""

    t0 = time.perf_counter()
    http_t0: float | None = None
    http_t1: float | None = None
    http_status: int | None = None
    row_count: int | None = None
    points_n: int | None = None
    out: str | None = None

    cc = (country_code or "US").strip().upper()[:2] or "US"
    q = urllib.parse.urlencode({"format": "json", "per_page": 25, "date": "2000:2025"})
    url = f"{_API}/{cc}/indicator/{_INDICATOR}?{q}"

    try:
        http_t0 = time.perf_counter()
        try:
            data, http_status = _http_get_json(url)
        except urllib.error.HTTPError as e:
            http_status = e.code
            out = f"World Bank API HTTP {e.code} for country {cc!r}."
            return out
        except urllib.error.URLError as e:
            out = f"World Bank API network error: {e.reason!r}."
            return out
        except (json.JSONDecodeError, OSError, ValueError) as e:
            out = f"World Bank API error: {e}"
            return out
        finally:
            http_t1 = time.perf_counter()

        if not isinstance(data, list) or len(data) < 2:
            out = "World Bank API: unexpected response shape."
            return out

        rows = data[1]
        if not isinstance(rows, list):
            out = "World Bank API: no observation list."
            return out

        row_count = len(rows)
        points: list[tuple[str, str]] = []
        country_label = cc
        indicator_label = _INDICATOR

        for row in rows:
            if not isinstance(row, dict) or row.get("value") is None:
                continue
            if "country" in row and isinstance(row["country"], dict):
                country_label = row["country"].get("value", country_label)
            if "indicator" in row and isinstance(row["indicator"], dict):
                indicator_label = row["indicator"].get("value", indicator_label)
            points.append((str(row.get("date", "")), str(row["value"])))

        points.sort(key=lambda x: x[0], reverse=True)
        if not points:
            out = f"No CPI inflation values returned for {cc} (check country code)."
            return out

        points_n = min(15, len(points))
        lines = [
            f"Source: World Bank API — {indicator_label}",
            f"Country: {country_label} ({cc})",
            "Annual % (recent years, newest first):",
        ]
        for d, v in points[:15]:
            lines.append(f"  {d}: {v}%")

        out = "\n".join(lines)
        return out
    finally:
        attrs: dict[str, object] = {
            "stats_api.country_code": cc,
            "stats_api.request_url": url,
            "stats_api.indicator_id": _INDICATOR,
            "stats_api.duration_ms": round((time.perf_counter() - t0) * 1000, 3),
        }
        if http_t0 is not None and http_t1 is not None:
            attrs["stats_api.http_duration_ms"] = round((http_t1 - http_t0) * 1000, 3)
        if http_status is not None:
            attrs["stats_api.http_status"] = http_status
        if row_count is not None:
            attrs["stats_api.observation_rows_in_response"] = row_count
        if points_n is not None:
            attrs["stats_api.points_in_summary"] = points_n
        if out is not None:
            attrs["stats_api.output_char_count"] = len(out)
            attrs["stats_api.output_line_count"] = out.count("\n") + 1
        _set_span_attrs(attrs)
