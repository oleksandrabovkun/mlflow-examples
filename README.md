# MLflow Agentic Examples

Hands-on examples showing how to observe and trace agentic AI systems with [MLflow](https://mlflow.org/).

## Examples

| Example | Framework | What you'll learn |
|---------|-----------|-------------------|
| [CrewAI Observability](./mlflow-crewai-observability/) | CrewAI + OpenAI | Trace a multi-agent crew end-to-end — agent spans, tool calls, LLM token counts, and custom validation metrics |
| [CrewAI Governance Guardrails](./mlflow-crewai-guardrails/) | CrewAI + OpenAI | Add AI governance guardrails on top of a traced crew — loop detection, PII redaction, cost attribution, and compliance auditing |
| [Multi-Modal Tracing](./mlflow-multimodal-tracing/) | OpenAI (vision / audio) | Trace agents that handle images, audio, and video — artifact-reference pattern, stable IDs for binary inputs, SQL-queryable traces |

## Contributing

Have an agentic framework + MLflow example to share? PRs welcome — drop a new folder with its own README and dependencies.
