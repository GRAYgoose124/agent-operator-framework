"""Real-time agent performance evaluation, ranking, and guidance generation."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from aof.agent.base import AgentMetrics, BaseAgent
from aof.config import EvaluatorConfig
from aof.inference.backend import InferenceBackend

logger = logging.getLogger(__name__)


@dataclass
class CanaryResult:
    """Results from a model capability check."""

    math_ok: bool = False
    json_ok: bool = False
    instruction_ok: bool = False
    overall_score: float = 0.0
    details: dict = field(default_factory=dict)


class AgentEvaluator:
    """Monitors agent health, ranks performance, and generates corrective guidance.

    Runs as a background task in the agent pool, periodically checking
    agent metrics and injecting tips into underperforming agents' prompts.
    """

    def __init__(self, config: EvaluatorConfig) -> None:
        self.config = config
        self._last_guidance: dict[str, float] = {}  # agent_id -> last guidance time

    def rank_agents(
        self, agents: list[BaseAgent]
    ) -> list[tuple[str, float, str]]:
        """Rank agents by health score.

        Returns list of (agent_id, health_score, role) sorted worst-first.
        """
        ranked = []
        for agent in agents:
            score = agent.ctx.metrics.health_score
            ranked.append((agent.ctx.agent_id, score, agent.ctx.role))
        ranked.sort(key=lambda x: x[1])
        return ranked

    def generate_guidance(self, agent: BaseAgent) -> str | None:
        """Generate corrective guidance for an underperforming agent.

        Returns a short instruction to inject into the agent's next system prompt,
        or None if the agent is healthy or was recently guided.
        """
        metrics = agent.ctx.metrics
        agent_id = agent.ctx.agent_id

        # Check health threshold
        if metrics.health_score >= self.config.health_threshold:
            return None

        # Cooldown: don't spam guidance
        now = time.monotonic()
        last = self._last_guidance.get(agent_id, 0.0)
        if now - last < self.config.guidance_cooldown_seconds:
            return None

        # Diagnose the primary issue and generate targeted guidance
        guidance = self._diagnose(metrics)
        if guidance:
            self._last_guidance[agent_id] = now
            logger.info(
                "Guidance for agent %s (health=%.2f): %s",
                agent_id,
                metrics.health_score,
                guidance,
            )

        return guidance

    def should_suggest_model_change(self, agent: BaseAgent) -> str | None:
        """Suggest a model tier upgrade if the agent's current model is inadequate.

        Returns a suggested role/tier string, or None.
        Model tiers (Qwen3 family):
          - micro: Qwen3-0.6B (basic search, simple tasks)
          - small: Qwen3-4B (code generation, moderate reasoning)
          - medium: Qwen3-8B+ (orchestration, complex planning)
        Model tiers (LFM2.5 family):
          - fast: LFM2.5-1.2B-Instruct (tool calling, agentic tasks, RAG)
          - reasoning: LFM2.5-1.2B-Thinking (math, chain-of-thought, planning)
        Cross-family suggestions:
          - micro struggling → fast (LFM2.5 excels at structured output)
          - fast struggling with reasoning → reasoning (thinking variant)
          - medium/reasoning struggling → fallback (Qwen3 14B for best performance)
        """
        metrics = agent.ctx.metrics
        role = agent.ctx.role

        # If agent consistently fails at planning or complex tool use
        if metrics.steps_attempted >= 3 and metrics.completion_rate < 0.3:
            if role == "micro":
                return "fast"  # LFM2.5-Instruct may handle it better
            if role == "fast":
                return "reasoning"  # Switch to thinking variant
            if role == "small":
                return "medium"
            if role in ("reasoning", "medium"):
                return "fallback"  # Escalate to Qwen3 14B

        # High parse failure rate suggests model can't handle structured output
        if metrics.steps_attempted >= 3:
            parse_rate = metrics.parse_failures / max(metrics.steps_attempted, 1)
            if parse_rate > 0.5:
                if role == "micro":
                    return "fast"  # LFM2.5 is better at structured output
                if role == "fast":
                    return "small"  # Larger Qwen3 may help
                if role in ("small", "reasoning"):
                    return "fallback"  # Qwen3 14B for best structured output

        return None

    async def run_canary(self, backend: InferenceBackend) -> CanaryResult:
        """Quick capability check for a model.

        Tests:
          1. Math: "What is 7 * 8?" -> expects "56"
          2. JSON output: "Output JSON with key 'color' value 'blue'"
          3. Instruction following: "Respond with only the word 'hello'"

        Handles Qwen3 thinking models that wrap output in <think>...</think> tags.
        """
        from aof.inference.parsing import detect_model_family, extract_json, parse_response

        result = CanaryResult()

        # Thinking models need extra tokens for reasoning overhead.
        # LFM2.5-Thinking generates much longer chains than Qwen3-0.6B.
        model_path = backend.model_info().get("model_path", backend.model_info().get("model", ""))
        is_thinking = "thinking" in model_path.lower()
        think_overhead = 512 if is_thinking else 256

        # Math test
        try:
            resp = await backend.complete(
                [
                    {"role": "system", "content": "Answer concisely. No explanation needed."},
                    {"role": "user", "content": "What is 7 * 8? Answer with just the number."},
                ],
                temperature=0.0,
                max_tokens=16 + think_overhead,
            )
            # Strip thinking tags before checking
            clean = parse_response(resp.text).text
            result.math_ok = "56" in clean
            result.details["math_response"] = clean.strip()
            result.details["math_raw"] = resp.text.strip()
        except Exception as e:
            result.details["math_error"] = str(e)

        # JSON test
        try:
            resp = await backend.complete(
                [
                    {"role": "system", "content": "Respond with only valid JSON. No explanation."},
                    {"role": "user", "content": 'Output a JSON object with key "color" and value "blue".'},
                ],
                temperature=0.0,
                max_tokens=64 + think_overhead,
            )
            clean = parse_response(resp.text).text
            try:
                parsed = extract_json(clean)
                result.json_ok = parsed.get("color") == "blue"
            except Exception:
                result.json_ok = False
            result.details["json_response"] = clean.strip()
            result.details["json_raw"] = resp.text.strip()
        except Exception as e:
            result.details["json_error"] = str(e)

        # Instruction following test
        try:
            resp = await backend.complete(
                [
                    {"role": "system", "content": "Follow instructions exactly."},
                    {"role": "user", "content": "Respond with only the word 'hello'. Nothing else."},
                ],
                temperature=0.0,
                max_tokens=8 + think_overhead,
            )
            clean = parse_response(resp.text).text.strip().lower().strip("'\".")
            result.instruction_ok = "hello" in clean and len(clean) < 20
            result.details["instruction_response"] = clean
            result.details["instruction_raw"] = resp.text.strip()
        except Exception as e:
            result.details["instruction_error"] = str(e)

        # Overall score
        passed = sum([result.math_ok, result.json_ok, result.instruction_ok])
        result.overall_score = passed / 3.0

        logger.info(
            "Canary results: math=%s, json=%s, instruction=%s (%.1f%%)",
            result.math_ok, result.json_ok, result.instruction_ok,
            result.overall_score * 100,
        )
        return result

    # -- private ------------------------------------------------------------

    def _diagnose(self, metrics: AgentMetrics) -> str | None:
        """Identify the primary issue and return targeted guidance."""

        # High parse failure rate
        if metrics.steps_attempted > 0:
            parse_rate = metrics.parse_failures / metrics.steps_attempted
            if parse_rate > 0.4:
                return (
                    "Your recent outputs had formatting issues. "
                    "When using a tool, respond with ONLY the tool_call JSON. "
                    "For other responses, use plain text only."
                )

        # Low tool accuracy
        if metrics.tool_calls_made >= 2 and metrics.tool_accuracy < 0.5:
            return (
                "Several tool calls failed. Before calling a tool, verify: "
                "1) the tool name is exact, 2) all required parameters are provided, "
                "3) parameter types match the schema."
            )

        # Low completion rate
        if metrics.steps_attempted >= 2 and metrics.completion_rate < 0.5:
            return (
                "Some steps are failing. Try breaking complex steps into simpler ones. "
                "Focus on one small action at a time."
            )

        # General low health
        if metrics.health_score < 0.5:
            return (
                "Focus on accuracy over speed. "
                "Read the task carefully and respond concisely."
            )

        return None
