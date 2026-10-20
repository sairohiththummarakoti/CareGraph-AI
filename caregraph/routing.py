"""Deterministic, bounded domain selection; no clinical inference."""

import re

ONCOLOGY = re.compile(
    r"\b(?:oncolog\w*|cancer\w*|tumou?r\w*|carcinoma\w*|sarcoma\w*|"
    r"lymphoma\w*|leuk[ae]mia\w*|chemotherap\w*|patholog\w*|biomarker\w*|metasta\w*|"
    r"radiation therapy|radiotherap\w*)\b",
    re.I,
)
CARDIOLOGY = re.compile(
    r"\b(?:cardi\w*|heart|myocardial|infarction|coronary|angina|"
    r"arrhythmia\w*|atrial fibrillation|atrial flutter|ventricular fibrillation|"
    r"atherosclero\w*|blood pressure|hypertension|chest pain|ecg|ekg)\b",
    re.I,
)
REPORT_CONTEXT = re.compile(r"\b(?:report|profile|findings|specimen|biopsy|these results)\b", re.I)


def classify_domains(question: str, confirmed_facts: dict[str, str] | None = None) -> list[str]:
    """A confirmed tumor report supplies oncology context, never a diagnosis."""
    domains: list[str] = []
    if ONCOLOGY.search(question) or (confirmed_facts and REPORT_CONTEXT.search(question)):
        domains.append("oncology")
    if CARDIOLOGY.search(question):
        domains.append("cardiology")
    return domains


def route_name(domains: list[str]) -> str:
    return "both" if len(domains) == 2 else domains[0] if domains else "fallback"
