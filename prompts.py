# -*- coding: utf-8 -*-
"""
prompts.py
==========
Reusable prompt blocks for stages 9-10.

Содержит:
• SYSTEM_PROMPT – инструкции модели
• JSON_SCHEMA   – «словесная» Pydantic-схема для жёсткого JSON-вывода
• build_final_prompt() – склеивает system + schema + context + question
"""
from __future__ import annotations

from textwrap import dedent


SYSTEM_PROMPT = dedent(
    """
    You are an assistant that answers questions ONLY with verifiable facts from
    the provided passages. If the answer cannot be found, respond with
    `"final_answer": "N/A"`. Always think step-by-step and cite the passage
    numbers you used.
    """
).strip()

JSON_SCHEMA = dedent(
    """
    You MUST produce a valid JSON object with the following keys:

    step_by_step_analysis: str  # your detailed reasoning (can reference pages)
    reasoning_summary: str      # 1-2 sentences TL;DR
    citations: List[int]        # page numbers you quoted (e.g. [2, 5])
    final_answer: str           # concise answer OR "N/A"
    """
).strip()


def build_final_prompt(context: str, question: str) -> str:
    """Concatenate all blocks into the final prompt sent to the LLM."""
    return "\n\n".join(
        [
            SYSTEM_PROMPT,
            "----- JSON SCHEMA -----",
            JSON_SCHEMA,
            "----- CONTEXT -----",
            context,
            "----- QUESTION -----",
            question.strip(),
            "",
            "Remember: output ONLY the JSON object and nothing else.",
        ]
    )
