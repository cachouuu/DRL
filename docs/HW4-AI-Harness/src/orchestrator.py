"""Simple controller workflow for the HW4 AI harness demo.

This file represents the orchestration layer. In a full AI system, an LLM would
decide which function to call next. In this mock demo, the decisions are written
as normal Python control flow so the workflow is easy to inspect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tools import (
    evaluate_presentation_readiness,
    explain_concept,
    generate_slide_outline,
    parse_paper_pdf,
    summarize_section,
)


@dataclass
class UserRequest:
    """User input to the AI harness."""

    pdf_path: str
    audience_level: str = "undergraduate"
    presentation_length_minutes: int = 8


@dataclass
class HarnessController:
    """Mock LLM controller with simple memory and tool-call tracing."""

    memory: dict[str, Any] = field(default_factory=dict)
    tool_trace: list[dict[str, str]] = field(default_factory=list)

    def run(self, request: UserRequest) -> dict[str, Any]:
        """Run the paper-reading and presentation workflow."""
        self.memory["user_request"] = request

        # Function call 1: parse the paper before any summarization can happen.
        parsed_paper = self._call_tool(
            "parse_paper_pdf",
            "Extract metadata, sections, and key concepts from the uploaded paper.",
            lambda: parse_paper_pdf(request.pdf_path),
        )
        self.memory["parsed_paper"] = parsed_paper

        # Function call 2: summarize each parsed paper section.
        section_summaries = {}
        for section_title, section_text in parsed_paper["sections"].items():
            section_summaries[section_title] = self._call_tool(
                "summarize_section",
                f"Summarize the {section_title} section.",
                lambda title=section_title, text=section_text: summarize_section(title, text),
            )
        self.memory["section_summaries"] = section_summaries

        # Function call 3: explain key concepts at the requested audience level.
        concept_explanations = [
            self._call_tool(
                "explain_concept",
                f"Explain '{concept}' for a {request.audience_level} audience.",
                lambda item=concept: explain_concept(item, request.audience_level),
            )
            for concept in parsed_paper["key_concepts"]
        ]
        self.memory["concept_explanations"] = concept_explanations

        # Function call 4: use summaries as input to generate a slide outline.
        slide_outline = self._call_tool(
            "generate_slide_outline",
            "Create a slide outline from section summaries and presentation length.",
            lambda: generate_slide_outline(
                section_summaries,
                request.presentation_length_minutes,
            ),
        )
        self.memory["slide_outline"] = slide_outline

        # Function call 5: evaluate the generated outline with a simple rubric.
        rubric = {
            "coverage": 30,
            "clarity": 25,
            "timing": 20,
            "takeaway": 25,
            "max_minutes": request.presentation_length_minutes,
        }
        evaluation = self._call_tool(
            "evaluate_presentation_readiness",
            "Check whether the slide outline is ready for presentation practice.",
            lambda: evaluate_presentation_readiness(slide_outline, rubric),
        )
        self.memory["evaluation"] = evaluation

        return {
            "metadata": parsed_paper["metadata"],
            "summary": section_summaries,
            "concept_explanations": concept_explanations,
            "slide_outline": slide_outline,
            "evaluation": evaluation,
            "tool_trace": self.tool_trace,
            "memory": self.memory,
        }

    def _call_tool(self, tool_name: str, reason: str, tool_call):
        """Record a tool call and execute it.

        This helper is the mock version of function calling: the controller
        chooses a tool, passes arguments through the callable, receives a
        structured result, and stores the result in memory.
        """
        self.tool_trace.append({"tool": tool_name, "reason": reason})
        return tool_call()
