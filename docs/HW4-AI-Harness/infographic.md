# HW4 Infographic: AI Paper Reading and Presentation Assistant

This infographic summarizes the proposed AI harness system for helping a student read a technical research paper and prepare a presentation. The system is a design/mock prototype: it defines the architecture, tools, memory, and orchestration logic, but it is not presented as a fully deployed product.

## 1. AI System Architecture

```mermaid
flowchart LR
    user["User<br/>Student presenter"]
    paper["Research paper PDF<br/>plus preferences"]
    controller["LLM Controller<br/>reasoning and supervision"]
    planner["Task Planner<br/>breaks goal into steps"]
    router["Tool Router<br/>selects function calls"]
    memory["Memory<br/>paper state, user preferences,<br/>feedback history"]
    evaluation["Evaluation Module<br/>readiness rubric and revision advice"]
    outputs["Final Outputs<br/>structured summary<br/>concept explanations<br/>slide outline<br/>speaking script<br/>readiness score"]

    subgraph tools["Tools"]
        pdf["parse_paper_pdf"]
        summarize["summarize_section"]
        explain["explain_concept"]
        slides["generate_slide_outline"]
        script["generate_speaking_script"]
        readiness["evaluate_presentation_readiness"]
    end

    user --> paper
    paper --> controller
    controller --> planner
    planner --> router
    router --> tools
    tools --> controller
    controller <--> memory
    controller --> evaluation
    evaluation --> controller
    controller --> outputs
```

**Explanation:** The LLM controller is the central decision maker. It receives the user goal, asks the task planner to break the work into steps, uses the tool router to call specialized tools, stores intermediate artifacts in memory, and sends the final materials through an evaluation module before returning them to the student.

## 2. Agent Workflow

```mermaid
flowchart TD
    start["User uploads paper"]
    prefs["User provides preferences<br/>audience level, talk length, goal"]
    plan["LLM plans tasks"]
    parse["PDF parser extracts structure<br/>title, abstract, sections, references"]
    summary["Section summarizer summarizes<br/>motivation, method, results, limitations"]
    concepts["Concept explainer explains terms<br/>methods, metrics, acronyms"]
    outline["Slide outline generator creates outline"]
    script["Speaking script generator drafts notes"]
    eval["Evaluation tool checks readiness"]
    decision{"Ready for presentation?"}
    revise["Controller revises weak part"]
    final["Final report is generated<br/>summary, concepts, outline, script, score"]

    start --> prefs
    prefs --> plan
    plan --> parse
    parse --> summary
    summary --> concepts
    concepts --> outline
    outline --> script
    script --> eval
    eval --> decision
    decision -- "Yes" --> final
    decision -- "No" --> revise
    revise --> eval
```

**Explanation:** The workflow moves from paper ingestion to presentation readiness. The system first builds a structured understanding of the paper, then generates presentation artifacts, then evaluates whether the result is strong enough. If the evaluation identifies a weak component, the controller loops back and revises it.

## 3. Function Calling / Tool Chain

```mermaid
sequenceDiagram
    participant U as User
    participant C as LLM Controller
    participant P as Task Planner
    participant R as Tool Router
    participant T as Tools
    participant M as Memory
    participant E as Evaluation Module

    U->>C: Upload PDF and request presentation help
    C->>M: Store user preferences
    C->>P: Plan paper-reading workflow
    P-->>C: Ordered task list
    C->>R: Request parse_paper_pdf(pdf)
    R->>T: parse_paper_pdf(pdf_file)
    T-->>R: paper_text, metadata, section_map
    R-->>C: Parsed paper structure
    C->>M: Save section_map

    C->>R: Request summarize_section(section_text)
    R->>T: summarize_section for major sections
    T-->>R: section summaries
    R-->>C: Structured paper summary
    C->>M: Save summaries

    C->>R: Request explain_concept(concept, context)
    R->>T: explain_concept for key terms
    T-->>R: concept explanations
    R-->>C: Explanations matched to paper context

    C->>R: Request generate_slide_outline(summary, preferences)
    R->>T: generate_slide_outline
    T-->>R: slide outline
    R-->>C: Slide plan

    C->>R: Request generate_speaking_script(outline)
    R->>T: generate_speaking_script
    T-->>R: speaker notes
    R-->>C: Speaking script

    C->>E: evaluate_presentation_readiness(outputs)
    E-->>C: readiness score and revision advice
    C->>M: Save evaluation feedback
    C-->>U: Return final report and presentation package
```

**Explanation:** This sequence diagram shows when each function is called and what data is passed. The controller does not directly generate every artifact in one step. It calls tools through the router, stores results in memory, and uses the evaluation module to decide whether the presentation package is ready.

## Visual Summary

- **Controller role:** plan, call tools, inspect outputs, revise weak artifacts.
- **Tool role:** perform specialized operations with explicit inputs and outputs.
- **Memory role:** preserve paper context, user preferences, intermediate drafts, and feedback.
- **Evaluation role:** check accuracy, coverage, clarity, slide quality, and delivery readiness.
- **Final output:** a structured paper-reading and presentation package for the student.
