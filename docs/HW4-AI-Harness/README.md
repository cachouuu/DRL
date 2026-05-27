# HW4: AI Harness System Design and Analysis

This GitHub Pages project presents an AI harness system for an **AI Paper Reading and Presentation Assistant**.

The project focuses on system design rather than model training. It describes how an LLM can act as a controller that coordinates tools, memory, workflow steps, and evaluation logic to help a student read a technical research paper and prepare a presentation.

## Deliverables

- [Project page](./index.html)
- [Written report](./report.md)
- [Infographic notes](./infographic.md)
- [AI-assisted design log](./log.md)
- [Architecture diagram](./diagrams/architecture.mmd)
- [Workflow diagram](./diagrams/workflow.mmd)
- [Sequence diagram](./diagrams/sequence.mmd)

## Application Topic

The assistant accepts a research paper PDF and produces:

- Structured paper summary
- Concept explanations
- Slide outline
- Speaking script
- Presentation readiness evaluation

## Current Implementation Scope

This folder contains a design scaffold and a runnable mock Python demo. The demo uses fake paper content and deterministic local functions to show how an AI harness can route tool calls, store intermediate results, and evaluate an output.

Complex PDF parsing, real LLM API calls, persistent vector memory, and slide deck export are intentionally left for later iterations.

## Run the Mock Demo

From the repository root:

```bash
cd docs/HW4-AI-Harness/src
python3 demo.py
```

The demo prints:

- Parsed paper metadata
- Structured section summaries
- Concept explanations
- Slide outline
- Presentation readiness evaluation
- Tool call trace showing the mock orchestration sequence

No external APIs, private keys, credentials, or network access are required.
