"""Run the HW4 mock AI harness demo.

Usage:
    cd docs/HW4-AI-Harness/src
    python3 demo.py
"""

from __future__ import annotations

from pprint import pprint

from orchestrator import HarnessController, UserRequest


def main() -> None:
    request = UserRequest(
        pdf_path="fake-paper.pdf",
        audience_level="undergraduate",
        presentation_length_minutes=8,
    )

    controller = HarnessController()
    result = controller.run(request)

    print("\n=== Parsed Metadata ===")
    pprint(result["metadata"])

    print("\n=== Structured Section Summary ===")
    for section, summary in result["summary"].items():
        print(f"\n[{section}]")
        print(summary["summary"])
        print("Key points:")
        for point in summary["key_points"]:
            print(f"- {point}")

    print("\n=== Concept Explanations ===")
    for item in result["concept_explanations"]:
        print(f"- {item['concept']} ({item['audience_level']}): {item['explanation']}")

    print("\n=== Slide Outline ===")
    for slide in result["slide_outline"]:
        print(f"\nSlide {slide['slide']}: {slide['title']} ({slide['minutes']} min)")
        for bullet in slide["bullets"]:
            print(f"- {bullet}")

    print("\n=== Evaluation Result ===")
    pprint(result["evaluation"])

    print("\n=== Tool Call Trace ===")
    for step, call in enumerate(result["tool_trace"], start=1):
        print(f"{step}. {call['tool']}: {call['reason']}")


if __name__ == "__main__":
    main()
