"""CareGraph's local command-line interface."""

import argparse
import json
from pathlib import Path

from caregraph.graph import run_workflow
from caregraph.parsing import MAX_REPORT_BYTES, ReportError, parse_report
from caregraph.retrieval import DEFAULT_INDEX_DIR


def _show_report(parsed: dict) -> None:
    print("\nExtracted synthetic report fields (not independently verified):")
    for field, value in parsed["pathology_facts"].items():
        print(f"  {field.replace('_', ' ')}: {value}")
    print("  Unknown / not provided: " + ", ".join(parsed["missing_facts"]))


def _print_result(result: dict) -> None:
    print("\n" + result.get("final_answer", "No answer was produced."))
    print(f"\nRoute: {result.get('route', 'pending')} | Assurance: {result.get('decision', 'pending')}")
    print("NODE = DO | EDGE = GO | STATE = KNOW")
    for event in result.get("trace", []):
        after = event["after"]
        print(f"  {event['node']} | changed: {', '.join(event['changed_keys']) or '(none)'} | evidence: {after['evidence_count']}")
        print("    before: " + json.dumps(event["before"], ensure_ascii=False))
        print("    after:  " + json.dumps(after, ensure_ascii=False))
    sources = result.get("evidence", [])
    if sources:
        print("\nSources:")
        for chunk in sources:
            print(f"  {chunk['source']} | {chunk['section']} | v{chunk['version']} | {chunk['chunk_id']}")
    print("Cloud tracing: " + result.get("cloud_trace_status", "disabled"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local, synthetic oncology/cardiology education; no paid LLM required.")
    parser.add_argument("--question", "-q", help="Educational question. Omit for the interactive prompt.")
    parser.add_argument("--report", type=Path, help="Synthetic UTF-8 TXT or text-based PDF, held in memory only.")
    parser.add_argument("--confirm-report", action="store_true", help="Explicitly confirm the extracted fields after reviewing a prior preview.")
    parser.add_argument("--json", action="store_true", help="Print structured results to stdout.")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    args = parser.parse_args(argv)
    data = None
    if args.report:
        try:
            if args.report.stat().st_size > MAX_REPORT_BYTES:
                raise ReportError("The report exceeds the 2 MiB limit.")
            data = args.report.read_bytes()
            parsed = parse_report(data, args.report.name)
        except (OSError, ReportError) as exc:
            result = {"error": str(exc), "decision": "block", "final_answer": str(exc)}
            print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else str(exc))
            return 2
        if not args.json:
            _show_report(parsed)
    question = args.question
    interactive = question is None and not args.json
    if interactive:
        print("CareGraph AI · synthetic education only")
        print("No diagnosis, prescribing, staging, or treatment selection. Type quit to exit.")
    while True:
        if interactive:
            try:
                question = input("\nQuestion: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if question.lower() in {"quit", "exit"}:
                return 0
        confirmed = args.confirm_report
        if data and not confirmed and interactive:
            try:
                confirmed = input("Have you reviewed the displayed synthetic report fields and unknowns? Type CONFIRM: ").strip() == "CONFIRM"
            except (EOFError, KeyboardInterrupt):
                return 0
        result = run_workflow(question or "", report_bytes=data, report_name=args.report.name if args.report else None, report_confirmed=confirmed, index_dir=args.index_dir)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            _print_result(result)
            if result.get("awaiting_confirmation"):
                print("\nAfter reviewing these fields, re-run with --confirm-report to permit retrieval.")
        if not interactive:
            return 2 if result.get("error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
