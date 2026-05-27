"""Mock tools for the HW4 AI harness demo.

The functions in this file represent external tools that an LLM controller
could call. They do not use real PDFs or external APIs; they return deterministic
mock data so the system design can be demonstrated locally.
"""

from __future__ import annotations


def parse_paper_pdf(pdf_path: str) -> dict[str, object]:
    """Mock PDF parser.

    Input:
        pdf_path: Path to a paper PDF.

    Output:
        Metadata, section text, and key concepts extracted from the paper.
    """
    return {
        "pdf_path": pdf_path,
        "metadata": {
            "title": "Harnessing Large Language Models for Research Assistance",
            "authors": ["A. Student", "B. Researcher"],
            "year": 2026,
            "venue": "Mock Conference on AI Systems",
        },
        "sections": {
            "Abstract": (
                "This paper studies how large language models can coordinate tools "
                "to help users complete research reading and presentation tasks."
            ),
            "Introduction": (
                "Students often struggle to convert dense technical papers into "
                "clear explanations. The paper proposes an AI harness that breaks "
                "the work into parsing, summarization, explanation, and evaluation."
            ),
            "Method": (
                "The method uses an LLM controller, a task planner, a tool router, "
                "short-term memory, and rubric-based evaluation to manage a multi-step workflow."
            ),
            "Results": (
                "In a mock evaluation, the harness improves organization, reduces "
                "missing presentation sections, and makes revision decisions easier to inspect."
            ),
            "Conclusion": (
                "The paper concludes that orchestration and evaluation are critical "
                "for reliable LLM-based research assistance."
            ),
        },
        "key_concepts": [
            "LLM controller",
            "tool routing",
            "short-term memory",
            "readiness evaluation",
        ],
    }


def summarize_section(section_title: str, section_text: str) -> dict[str, object]:
    """Mock section summarizer.

    In a real system this could be an LLM call. Here, it compresses each section
    into a short summary and simple key points.
    """
    first_sentence = section_text.split(".")[0].strip()
    return {
        "section": section_title,
        "summary": first_sentence + ".",
        "key_points": _key_points_for_section(section_title),
    }


def explain_concept(concept: str, audience_level: str) -> dict[str, str]:
    """Mock concept explainer."""
    explanations = {
        "LLM controller": (
            "An LLM controller is the central decision maker that chooses which "
            "tool to call next and checks whether the result is useful."
        ),
        "tool routing": (
            "Tool routing means selecting the right function for the current task, "
            "such as parsing a PDF before summarizing its sections."
        ),
        "short-term memory": (
            "Short-term memory stores the current paper, intermediate summaries, "
            "and feedback during one workflow."
        ),
        "readiness evaluation": (
            "Readiness evaluation checks whether the generated presentation "
            "materials are accurate, complete, clear, and usable."
        ),
    }

    return {
        "concept": concept,
        "audience_level": audience_level,
        "explanation": explanations.get(
            concept,
            f"{concept} is an important idea explained for a {audience_level} audience.",
        ),
    }


def generate_slide_outline(
    paper_summary: dict[str, dict[str, object]],
    presentation_length_minutes: int,
) -> list[dict[str, object]]:
    """Mock slide outline generator."""
    return [
        {
            "slide": 1,
            "title": "Paper Motivation",
            "minutes": 1.5,
            "bullets": [
                paper_summary["Introduction"]["summary"],
                "Why students need structured help reading technical papers.",
            ],
        },
        {
            "slide": 2,
            "title": "System Architecture",
            "minutes": 2.0,
            "bullets": [
                paper_summary["Method"]["summary"],
                "Controller, planner, router, tools, memory, and evaluation.",
            ],
        },
        {
            "slide": 3,
            "title": "Mock Results",
            "minutes": 1.5,
            "bullets": [
                paper_summary["Results"]["summary"],
                "The workflow makes intermediate decisions inspectable.",
            ],
        },
        {
            "slide": 4,
            "title": "Takeaways",
            "minutes": max(1.0, presentation_length_minutes - 5.0),
            "bullets": [
                paper_summary["Conclusion"]["summary"],
                "Reliable AI systems require orchestration, tools, memory, and evaluation.",
            ],
        },
    ]


def evaluate_presentation_readiness(
    outline: list[dict[str, object]],
    rubric: dict[str, int],
) -> dict[str, object]:
    """Mock readiness evaluator.

    The score is intentionally simple: it rewards slide coverage and the presence
    of key presentation elements. This represents an evaluation tool call.
    """
    slide_titles = " ".join(str(slide["title"]).lower() for slide in outline)
    checks = {
        "coverage": all(term in slide_titles for term in ["motivation", "architecture", "results"]),
        "clarity": all(len(slide.get("bullets", [])) >= 2 for slide in outline),
        "timing": sum(float(slide.get("minutes", 0)) for slide in outline) <= rubric["max_minutes"],
        "takeaway": "takeaways" in slide_titles,
    }

    score = sum(rubric[name] for name, passed in checks.items() if passed)
    max_score = sum(rubric[name] for name in checks)
    recommendations = []
    if not checks["coverage"]:
        recommendations.append("Add slides that cover motivation, architecture, and results.")
    if not checks["clarity"]:
        recommendations.append("Add at least two clear bullets to each slide.")
    if not checks["timing"]:
        recommendations.append("Shorten the outline to fit the presentation length.")
    if not checks["takeaway"]:
        recommendations.append("Add a final takeaway slide.")
    if not recommendations:
        recommendations.append("The mock outline is ready for a first practice run.")

    return {
        "score": score,
        "max_score": max_score,
        "checks": checks,
        "recommendations": recommendations,
    }


def _key_points_for_section(section_title: str) -> list[str]:
    """Return small mock key points for a section title."""
    section_points = {
        "Abstract": [
            "The paper studies LLM-based research assistance.",
            "The central idea is tool coordination.",
        ],
        "Introduction": [
            "Technical papers are difficult to convert into presentations.",
            "A harness can make the reading workflow more structured.",
        ],
        "Method": [
            "The LLM acts as a controller.",
            "Tools, memory, and evaluation are separated into modules.",
        ],
        "Results": [
            "The mock workflow improves organization.",
            "Intermediate outputs make revision easier.",
        ],
        "Conclusion": [
            "System design matters as much as raw model output.",
            "Future work can add real PDF parsing and API-based LLM calls.",
        ],
    }
    return section_points.get(section_title, ["No mock key points available."])
