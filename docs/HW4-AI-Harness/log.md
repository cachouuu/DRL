# HW4 AI-Assisted Design and Development Log

This log documents the AI-assisted design process for HW4: **AI Harness System Design and Analysis**. The project is a design/mock prototype for an **AI Paper Reading and Presentation Assistant**. No private information is included, and no external APIs were actually called during the mock implementation.

## 1. Initial Understanding of the Assignment

The assignment required a system design project rather than a model-training project. The main focus was on AI harness concepts: an LLM as a controller, tool use/function calling, multi-step workflow, memory, orchestration, and evaluation.

Representative prompt:

```text
I am working on Homework 4 for my Deep Reinforcement Learning course.
The assignment is about AI Harness Systems Design and Analysis.
The focus is NOT model training. The focus is system design.
Please inspect the current repository structure and propose a clean folder structure,
files to create, files to modify, and an implementation plan.
```

Design decision:

- Treat HW4 as a GitHub Pages project under `docs/HW4-AI-Harness/`.
- Keep HW1, HW2, and HW3 unchanged.
- Include a written report, infographic, design log, Mermaid diagrams, and a small mock demo.

## 2. Topic Selection

The selected topic was **AI Paper Reading and Presentation Assistant for technical research papers**.

Representative prompt:

```text
Chosen application topic:
AI Paper Reading and Presentation Assistant for technical research papers.
The system should help a student read a research paper PDF and generate:
structured paper summary, concept explanations, slide outline, speaking script,
and presentation readiness evaluation.
```

Why this topic was chosen:

- It naturally requires a multi-step workflow.
- It demonstrates the LLM as a controller rather than only a text generator.
- It has clear tool boundaries: PDF parsing, summarization, concept explanation, slide planning, and evaluation.
- It supports memory design because the system must remember paper context, student preferences, and feedback.
- It is realistic for a student use case and fits the assignment's focus on AI system architecture.

## 3. First Architecture Design

The first architecture separated the system into layers:

- User interface
- LLM controller
- Task planner
- Tool router
- Tool layer
- Memory layer
- Evaluation module
- Final output layer

Representative prompt:

```text
Please propose an AI harness architecture for the paper reading assistant.
Include LLM controller, tools, memory, orchestration, evaluation, and data flow.
```

Initial design decision:

- The LLM controller should not generate everything in one response.
- The controller should decide which tool to call, pass data to that tool, store the result, and continue the workflow.
- Intermediate artifacts should be inspectable: parsed paper structure, section summaries, concept explanations, slide outline, and readiness score.

Architecture refinement:

- The design was changed from a general chatbot into a function-calling harness.
- A task planner and tool router were added to make the controller's responsibilities clearer.
- The evaluation module was made explicit so the system checks output quality before returning final materials.

## 4. Tool Design Iteration

The required tools were:

- `parse_paper_pdf`
- `summarize_section`
- `explain_concept`
- `generate_slide_outline`
- `evaluate_presentation_readiness`

Representative prompt:

```text
Please write the report section for Tool Design.
Clearly explain inputs, outputs, and when each tool is called.
Include at least five tools: parse_paper_pdf, summarize_section,
explain_concept, generate_slide_outline, and evaluate_presentation_readiness.
```

Design decisions:

- `parse_paper_pdf` is called first because all later steps depend on paper content.
- `summarize_section` is called separately for major sections to avoid losing paper structure.
- `explain_concept` is called for key terms and adapts explanations to the audience level.
- `generate_slide_outline` is called after summaries exist.
- `evaluate_presentation_readiness` is called after the presentation materials are generated.

Tool design improvement:

- Each tool was given a narrow responsibility.
- Inputs and outputs were kept structured so the controller can store results in memory.
- The tools were designed as mock functions in Python for demonstration, not as real external API calls.

## 5. Workflow Design Iteration

The workflow was designed as a sequence with a possible revision loop:

1. User uploads paper.
2. User provides preferences such as audience level and talk length.
3. LLM controller plans the task.
4. PDF parser extracts paper structure.
5. Section summarizer summarizes each section.
6. Concept explainer explains important terms.
7. Slide outline generator creates a presentation outline.
8. Evaluation tool checks readiness.
9. Controller revises weak parts if needed.
10. Final report and presentation package are returned.

Representative prompt:

```text
Please create the agent workflow for the AI Paper Reading and Presentation Assistant.
Show how the LLM plans tasks, calls tools, stores memory, evaluates readiness,
and revises weak outputs.
```

Workflow decision:

- The first version is mostly sequential for clarity.
- A revision loop was added after evaluation because first drafts may be incomplete.
- The workflow intentionally avoids claiming that the mock system can fully understand arbitrary PDFs.

## 6. Infographic Design Decisions

The infographic needed to visualize system architecture, workflow, and function calling/tool chain.

Representative prompt:

```text
Please create the infographic content for HW4.
The infographic should visualize AI system architecture, workflow,
and function calling/tool chain. Use Mermaid diagrams in Markdown.
```

Design decisions:

- Mermaid was used because it works well in Markdown and is suitable for GitHub Pages documentation.
- Three diagrams were created:
  - `architecture.mmd`: shows User, LLM Controller, Task Planner, Tool Router, Tools, Memory, Evaluation Module, and Final Outputs.
  - `workflow.mmd`: shows the step-by-step paper-to-presentation workflow.
  - `sequence.mmd`: shows the function-calling order and data passed between controller, tools, memory, and evaluation.
- Short explanations were added under each diagram to make the infographic understandable without extra context.

Infographic improvement:

- The HW4 `index.html` page was updated to include inline Mermaid diagrams, while the standalone `.mmd` files remain available as source files.

## 7. Mock Implementation Decisions

The implementation was kept simple and local. The goal was to demonstrate the harness concept without requiring real APIs, private keys, or complex dependencies.

Representative prompt:

```text
Please implement a simple mock demo for HW4 under docs/HW4-AI-Harness/src/.
Demonstrate the AI Harness concept without requiring real external APIs.
Implement tools.py, orchestrator.py, and demo.py.
```

Implementation decisions:

- `tools.py` contains deterministic mock tools.
- `orchestrator.py` contains a simple controller workflow.
- `demo.py` runs a fake paper workflow and prints outputs.
- A memory dictionary stores intermediate results.
- A tool trace records which mock tool was called and why.

The mock tools include:

- `parse_paper_pdf(pdf_path)`: returns fake paper metadata, sections, and concepts.
- `summarize_section(section_title, section_text)`: returns a short summary and key points.
- `explain_concept(concept, audience_level)`: returns audience-aware concept explanations.
- `generate_slide_outline(paper_summary, presentation_length_minutes)`: creates a mock slide outline.
- `evaluate_presentation_readiness(outline, rubric)`: returns a score, checks, and recommendations.

Important limitation:

- The code does not parse real PDFs.
- The code does not call any LLM API.
- The code does not use private keys, credentials, or external services.
- The demo is a mock representation of orchestration and function calling.

## 8. Problems Found and Fixes

Problem 1: The early design looked too much like a general chatbot.

Fix:

- Added explicit components: LLM Controller, Task Planner, Tool Router, Tools, Memory, and Evaluation Module.
- Added a function-calling sequence diagram to show tool usage clearly.

Problem 2: The first tool names were inconsistent with the report requirements.

Fix:

- Standardized the required tool names around `parse_paper_pdf`, `summarize_section`, `explain_concept`, `generate_slide_outline`, and `evaluate_presentation_readiness`.

Problem 3: The first implementation was only placeholder stubs.

Fix:

- Replaced stubs with a runnable mock workflow.
- Added fake paper content, section summaries, concept explanations, slide outline generation, evaluation result, memory storage, and tool tracing.

Problem 4: The mock readiness score initially calculated the maximum score incorrectly.

Fix:

- Updated the evaluator so only rubric criteria are counted in the maximum score.
- Verified that the demo now reports `score: 100` and `max_score: 100` for the complete mock outline.

Problem 5: Python compile checking attempted to write bytecode into a blocked macOS cache path.

Fix:

- Re-ran the syntax check with `PYTHONPYCACHEPREFIX=/private/tmp/hw4_pycache`.
- The code passed syntax compilation with that cache path.

Problem 6: Rendered browser verification was attempted, but the in-app browser backend was unavailable.

Fix:

- Performed static checks of the Markdown, Mermaid source files, HTML embedding, and demo execution.
- The project remains suitable for GitHub Pages, but final visual rendering should be checked in a normal browser if needed.

## 9. Final Design Summary

The final HW4 project presents an AI harness design for a paper reading and presentation assistant. The system is organized around an LLM controller that plans tasks, routes function calls, stores intermediate results in memory, and uses an evaluation module to judge readiness.

Final project components:

- `report.md`: written report explaining the system design.
- `infographic.md`: Mermaid-based infographic content.
- `diagrams/architecture.mmd`: architecture diagram.
- `diagrams/workflow.mmd`: workflow diagram.
- `diagrams/sequence.mmd`: function-calling sequence diagram.
- `src/tools.py`: mock tool implementations.
- `src/orchestrator.py`: simple controller and memory workflow.
- `src/demo.py`: runnable local demonstration.
- `index.html`: GitHub Pages project page.

The final design is logically consistent with the assignment because it emphasizes AI system design rather than model training. It demonstrates how an LLM can act as a controller in a larger harness that includes tools, memory, orchestration, evaluation, and final user-facing outputs.
