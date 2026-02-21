"""
Tool Registry for the ReAct Agent

Defines the tools that the LLM can autonomously select and invoke
during a ReAct reasoning loop.  Each tool wraps a domain-engine
method and exposes a name, description, parameter schema, and
callable.

This is the heart of the agentic pattern — the LLM decides WHICH
tool to call and with WHAT parameters, rather than the code
hard-coding the routing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ── Tool definition ───────────────────────────────────────────────────────

@dataclass
class Tool:
    """A single tool the agent can invoke.

    Attributes
    ----------
    name : str
        Machine-readable identifier the LLM uses to call this tool.
    description : str
        Human-readable description shown to the LLM so it can decide
        when to use this tool.
    parameters : dict
        JSON-Schema-like description of expected parameters.
    fn : Callable
        The actual function to execute.  Receives ``**kwargs`` matching
        the parameter schema and returns a ``dict``.
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    fn: Callable[..., dict[str, Any]] = field(default=lambda **kw: {})


# ── Tool registry ─────────────────────────────────────────────────────────

class ToolRegistry:
    """Container for all tools available to the agent.

    Usage::

        registry = ToolRegistry()
        registry.register(tool)
        result = registry.execute("tool_name", {"param": "value"})
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool (overwrites if name already exists)."""
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def execute(self, name: str, parameters: dict[str, Any] | None = None) -> dict[str, Any]:
        """Look up a tool by name and execute it.

        Returns
        -------
        dict
            Tool output on success, or an error dict if the tool is
            not found or raises.
        """
        tool = self._tools.get(name)
        if tool is None:
            return {
                "error": f"Unknown tool '{name}'.",
                "available_tools": list(self._tools.keys()),
            }

        params = parameters or {}
        try:
            result = tool.fn(**params)
            return result
        except Exception as exc:
            logger.warning("Tool '%s' raised: %s", name, exc)
            return {"error": f"Tool '{name}' failed: {exc}"}

    @property
    def names(self) -> list[str]:
        return list(self._tools.keys())

    def describe_for_llm(self) -> str:
        """Render a text block describing all tools for the system prompt.

        The format is designed to be unambiguous for the LLM so it can
        pick the right tool and construct valid parameters.
        """
        lines: list[str] = []
        for t in self._tools.values():
            lines.append(f"### {t.name}")
            lines.append(f"Description: {t.description}")
            if t.parameters:
                lines.append("Parameters (JSON object):")
                for pname, pinfo in t.parameters.items():
                    required = " (required)" if pinfo.get("required") else " (optional)"
                    desc = pinfo.get("description", "")
                    lines.append(f"  - {pname}{required}: {desc}")
            else:
                lines.append("Parameters: none (pass empty object {})")
            lines.append("")
        return "\n".join(lines)


# ── Factory: build registry from engines ──────────────────────────────────

def build_tool_registry(
    survival_engine,
    analytics_engine,
) -> ToolRegistry:
    """Create a fully-populated ToolRegistry from the domain engines.

    This is called once at agent initialisation.
    """
    registry = ToolRegistry()

    # ── Survival tools ────────────────────────────────────────────────

    def _model_summary(**_kw) -> dict[str, Any]:
        model = survival_engine.fitted_model
        summary_df = model.summary
        coefficients = []
        for idx, row in summary_df.iterrows():
            coefficients.append({
                "covariate": str(idx),
                "coefficient": round(float(row["coef"]), 4),
                "hazard_ratio": round(float(row["exp(coef)"]), 4),
                "p_value": round(float(row["p"]), 4),
                "significance": (
                    "significant" if float(row["p"]) < 0.05 else "not significant"
                ),
            })
        return {
            "tool": "get_survival_model_summary",
            "concordance_index": round(survival_engine.concordance_index, 4),
            "num_observations": int(model.summary.shape[0]),
            "coefficients": coefficients,
        }

    registry.register(Tool(
        name="get_survival_model_summary",
        description=(
            "Retrieve the Cox Proportional-Hazards model summary including "
            "concordance index, coefficients, hazard ratios, and p-values "
            "for all covariates."
        ),
        parameters={},
        fn=_model_summary,
    ))

    def _risk_factors(**_kw) -> dict[str, Any]:
        model = survival_engine.fitted_model
        summary_df = model.summary.copy()
        summary_df["abs_coef"] = summary_df["coef"].abs()
        summary_df = summary_df.sort_values("abs_coef", ascending=False)
        risk_factors = []
        for idx, row in summary_df.iterrows():
            direction = "increases" if float(row["coef"]) > 0 else "decreases"
            risk_factors.append({
                "factor": str(idx),
                "hazard_ratio": round(float(row["exp(coef)"]), 4),
                "direction": direction,
                "p_value": round(float(row["p"]), 4),
                "significant": bool(float(row["p"]) < 0.05),
            })
        return {
            "tool": "get_risk_factors",
            "risk_factors": risk_factors,
            "concordance_index": round(survival_engine.concordance_index, 4),
        }

    registry.register(Tool(
        name="get_risk_factors",
        description=(
            "Get the most influential risk factors for survival, sorted by "
            "effect size.  Returns hazard ratios, direction of effect, and "
            "statistical significance."
        ),
        parameters={},
        fn=_risk_factors,
    ))

    def _survival_prediction(**kwargs) -> dict[str, Any]:
        # If no features provided, ask for them
        if not kwargs:
            return {
                "tool": "predict_survival",
                "message": (
                    "Survival prediction requires specific patient features.  "
                    "Please provide: age_group, gleason_score, bone_metastasis "
                    "(Y/N), and positive_nodes (Y/N)."
                ),
            }
        return {
            "tool": "predict_survival",
            "message": (
                "Survival prediction requires specific patient features."
            ),
            "parameters_received": kwargs,
        }

    registry.register(Tool(
        name="predict_survival",
        description=(
            "Predict survival / hazard for a specific patient profile.  "
            "Requires patient features like age group, Gleason score, "
            "bone metastasis status, and positive node status."
        ),
        parameters={
            "age_group": {
                "description": "Patient age group, e.g. '60-64', '65-69', '70+'",
                "required": False,
            },
            "gleason_score": {
                "description": "Gleason score (integer)",
                "required": False,
            },
            "bone_metastasis": {
                "description": "Bone metastasis status: 'Y' or 'N'",
                "required": False,
            },
            "positive_nodes": {
                "description": "Positive lymph node status: 'Y' or 'N'",
                "required": False,
            },
        },
        fn=_survival_prediction,
    ))

    # ── Analytics tools ───────────────────────────────────────────────

    def _make_analytics_tool(intent: str):
        """Factory: wrap analytics_engine.query(intent, params)."""
        def _fn(**kwargs) -> dict[str, Any]:
            result = analytics_engine.query(intent, kwargs)
            result["tool"] = f"get_{intent}"
            return result
        return _fn

    registry.register(Tool(
        name="get_progression_stats",
        description=(
            "Get disease progression statistics: progression rate, median / "
            "mean / min / max days to progression."
        ),
        parameters={},
        fn=_make_analytics_tool("progression_stats"),
    ))

    registry.register(Tool(
        name="get_mortality_stats",
        description=(
            "Get mortality statistics: death rate, cause-of-death breakdown, "
            "and time-to-death distribution."
        ),
        parameters={},
        fn=_make_analytics_tool("mortality_stats"),
    ))

    registry.register(Tool(
        name="get_lab_summary",
        description=(
            "Get baseline lab-value distributions for PSA, bilirubin, AST, "
            "and ALT — including mean, median, standard deviation, and range."
        ),
        parameters={},
        fn=_make_analytics_tool("lab_summary"),
    ))

    registry.register(Tool(
        name="get_assessment_summary",
        description=(
            "Get assessment summary: prevalence of bone metastasis and "
            "positive lymph nodes at baseline."
        ),
        parameters={},
        fn=_make_analytics_tool("assessment_summary"),
    ))

    registry.register(Tool(
        name="get_gleason_distribution",
        description=(
            "Get the distribution of Gleason scores across the cohort."
        ),
        parameters={},
        fn=_make_analytics_tool("gleason_distribution"),
    ))

    registry.register(Tool(
        name="get_adverse_events",
        description=(
            "Get adverse-event rates, severity breakdown, and most common "
            "AEs — optionally broken down by age group or race."
        ),
        parameters={},
        fn=_make_analytics_tool("adverse_events_by_demographics"),
    ))

    logger.info(
        "Tool registry built with %d tools: %s",
        len(registry.names),
        registry.names,
    )
    return registry
