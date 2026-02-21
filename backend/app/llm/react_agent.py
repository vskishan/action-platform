"""
ReAct Agent — Agentic AI Orchestrator

An autonomous agent that uses the **ReAct** (Reason + Act) pattern
to answer clinical-trial research questions.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from backend.app.engine.analytics_engine import AnalyticsEngine
from backend.app.engine.survival_engine import SurvivalAnalysisEngine
from backend.app.llm.medgemma_client import MedGemmaClient
from backend.app.llm.memory import ConversationMemory
from backend.app.llm.prompts import REACT_SYSTEM_PROMPT, REACT_FOLLOW_UP_PROMPT
from backend.app.llm.tools import ToolRegistry, build_tool_registry

logger = logging.getLogger(__name__)

# Constants

MAX_REACT_STEPS = 5          # Maximum tool calls per query
REACT_TEMPERATURE = 0.1      # Low temperature for structured reasoning


class ReactAgent:
    """Agentic orchestrator using the ReAct pattern.

    Usage::

        agent = ReactAgent()

        # First query — creates a new session
        result = agent.handle("What are the risk factors for survival?")
        session_id = result["session_id"]

        # Follow-up — uses the same session
        result = agent.handle("How does that compare to mortality?",
                              session_id=session_id)
    """

    def __init__(self, model_name: str | None = None) -> None:
        self._client = MedGemmaClient.get_instance(model=model_name)
        self._memory = ConversationMemory.get_instance()

        # Initialise engines.
        logger.info("Initialising engines for ReAct agent...")
        self._survival_engine = SurvivalAnalysisEngine().run()
        self._analytics_engine = AnalyticsEngine().run()

        # Build the tool registry from engines.
        self._tools: ToolRegistry = build_tool_registry(
            survival_engine=self._survival_engine,
            analytics_engine=self._analytics_engine,
        )

        logger.info("ReAct agent is ready with %d tools.", len(self._tools.names))

    # Public API

    def handle(
        self,
        user_query: str,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Process a query using the ReAct loop.

        Parameters
        ----------
        user_query : str
            The user's natural-language question.
        session_id : str | None
            Optional session ID for conversation continuity.
            If ``None``, a new session is created.

        Returns
        -------
        dict
            {
                "query": <original query>,
                "response": <final answer>,
                "session_id": <session ID for follow-ups>,
                "tools_used": [list of tools called],
                "steps": <number of reasoning steps>,
            }
        """
        # Resolve session.
        session_id = self._memory.get_or_create_session(session_id)
        self._memory.add_user_message(session_id, user_query)

        # Gather conversation context.
        conversation_context = self._memory.get_context_for_prompt(session_id)

        # Determine if this is a follow-up (session has prior messages).
        prior_messages = self._memory.get_messages(session_id)
        is_follow_up = len([m for m in prior_messages if m.role == "user"]) > 1

        # Build the system prompt.
        system_prompt = REACT_SYSTEM_PROMPT.format(
            max_steps=MAX_REACT_STEPS,
            tools_description=self._tools.describe_for_llm(),
            conversation_context=conversation_context if conversation_context else "(No prior conversation)",
        )

        # Build the initial user prompt.
        if is_follow_up:
            user_prompt = REACT_FOLLOW_UP_PROMPT.format(
                conversation_context=conversation_context,
                user_query=user_query,
            )
        else:
            user_prompt = f"User question: {user_query}"

        # Run the ReAct loop.
        answer, tools_used, steps = self._react_loop(
            system_prompt=system_prompt,
            initial_prompt=user_prompt,
            session_id=session_id,
        )

        # Store the final answer in memory.
        self._memory.add_assistant_message(session_id, answer)

        logger.info(
            "ReAct completed in %d steps using tools: %s",
            steps,
            tools_used,
        )

        return {
            "query": user_query,
            "response": answer,
            "session_id": session_id,
            "tools_used": tools_used,
            "steps": steps,
        }

    # ReAct loop

    def _react_loop(
        self,
        system_prompt: str,
        initial_prompt: str,
        session_id: str,
    ) -> tuple[str, list[str], int]:
        """Execute the Thought → Action → Observation loop.

        Returns
        -------
        tuple[str, list[str], int]
            (final_answer, tools_used, step_count)
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial_prompt},
        ]

        tools_used: list[str] = []
        step = 0

        while step < MAX_REACT_STEPS:
            step += 1
            logger.info("ReAct step %d/%d", step, MAX_REACT_STEPS)

            # Ask the LLM for its next thought/action.
            llm_response = self._client.chat_messages(
                messages,
                temperature=REACT_TEMPERATURE,
            )

            logger.debug("LLM response (step %d): %s", step, llm_response[:300])

            # Parse the response for Action or Answer.
            action = self._parse_action(llm_response)
            answer = self._parse_answer(llm_response)

            if answer is not None:
                # Agent decided it has enough information.
                return answer, tools_used, step

            if action is not None:
                tool_name = action.get("tool", "")
                parameters = action.get("parameters", {})

                logger.info(
                    "Step %d — Agent calls tool: %s(%s)",
                    step,
                    tool_name,
                    parameters,
                )

                # Execute the tool.
                tool_result = self._tools.execute(tool_name, parameters)
                tools_used.append(tool_name)

                # Record in memory.
                self._memory.add_tool_call(session_id, tool_name, parameters)
                self._memory.add_tool_result(session_id, tool_name, tool_result)

                # Feed the observation back to the LLM.
                observation_text = (
                    f"Observation (from {tool_name}):\n"
                    f"{json.dumps(tool_result, indent=2, default=str)}"
                )

                # Add the LLM's response and the observation to messages.
                messages.append({"role": "assistant", "content": llm_response})
                messages.append({"role": "user", "content": observation_text})

            else:
                # LLM didn't produce a valid Action or Answer.
                # Nudge it to try again.
                logger.warning(
                    "Step %d — Could not parse Action or Answer. Nudging LLM.",
                    step,
                )
                messages.append({"role": "assistant", "content": llm_response})
                messages.append({
                    "role": "user",
                    "content": (
                        "I could not parse a valid Action or Answer from your "
                        "response.  Please respond with EXACTLY one of:\n"
                        "1. A Thought + Action (JSON with 'tool' and 'parameters')\n"
                        "2. A Thought + Answer (your final response)\n\n"
                        "Try again."
                    ),
                })

        # Reached max steps — ask LLM to synthesise a final answer.
        logger.warning(
            "ReAct reached max steps (%d). Forcing final answer.",
            MAX_REACT_STEPS,
        )
        messages.append({
            "role": "user",
            "content": (
                "You have reached the maximum number of tool calls.  "
                "Please synthesise the best possible Answer from the "
                "data you have gathered so far."
            ),
        })

        forced_response = self._client.chat_messages(
            messages,
            temperature=REACT_TEMPERATURE,
        )
        answer = self._parse_answer(forced_response)
        if answer:
            return answer, tools_used, step

        # Last resort — return the raw LLM text.
        return forced_response, tools_used, step

    # Parsing helpers

    @staticmethod
    def _parse_action(text: str) -> dict[str, Any] | None:
        """Extract a tool-call Action from the LLM's output.

        Looks for patterns like:
            Action: {"tool": "get_risk_factors", "parameters": {}}
        """
        # Pattern 1: Action: {json}
        action_match = re.search(
            r'Action:\s*(\{.*?\})\s*$',
            text,
            re.MULTILINE | re.DOTALL,
        )
        if action_match:
            try:
                parsed = json.loads(action_match.group(1))
                if "tool" in parsed:
                    return parsed
            except json.JSONDecodeError:
                pass

        # Pattern 2: standalone JSON block with "tool" key in code fence
        json_match = re.search(
            r'```(?:json)?\s*(\{[^`]*?"tool"[^`]*?\})\s*```',
            text,
            re.DOTALL,
        )
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
                if "tool" in parsed:
                    return parsed
            except json.JSONDecodeError:
                pass

        # Pattern 3: bare JSON on a line following "Action:"
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if line.strip().lower().startswith("action:"):
                # The JSON might be on the same line or the next line
                json_part = line.split("Action:", 1)[-1].strip()
                if not json_part and i + 1 < len(lines):
                    json_part = lines[i + 1].strip()

                # Try to parse everything from the first { to the last }
                brace_start = json_part.find("{")
                if brace_start >= 0:
                    # Gather remaining text
                    remaining = json_part[brace_start:]
                    for end_line in lines[i + 1:]:
                        remaining += "\n" + end_line
                        try:
                            parsed = json.loads(remaining[:remaining.rindex("}") + 1])
                            if "tool" in parsed:
                                return parsed
                        except (json.JSONDecodeError, ValueError):
                            continue

        return None

    @staticmethod
    def _parse_answer(text: str) -> str | None:
        """Extract a final Answer from the LLM's output.

        Looks for patterns like:
            Answer: <the final answer text>
        """
        # Pattern: Answer: <everything after>
        answer_match = re.search(
            r'^Answer:\s*(.+)',
            text,
            re.MULTILINE | re.DOTALL,
        )
        if answer_match:
            answer_text = answer_match.group(1).strip()
            # Make sure it's not also followed by an Action (partial answer)
            action_in_answer = re.search(r'^Action:', answer_text, re.MULTILINE)
            if action_in_answer:
                answer_text = answer_text[:action_in_answer.start()].strip()
            if answer_text:
                return answer_text

        return None

    # Convenience

    @property
    def available_tools(self) -> list[str]:
        """List of registered tool names."""
        return self._tools.names

    def get_session_history(self, session_id: str) -> list[dict[str, str]]:
        """Return conversation history for a session (for debugging)."""
        messages = self._memory.get_messages(session_id)
        return [
            {"role": m.role, "content": m.content, "timestamp": m.timestamp}
            for m in messages
        ]
