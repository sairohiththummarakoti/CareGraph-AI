"""Deterministic scope gates and a rewrite followed by revalidation."""

import re
from .synthesis import DISCLAIMER, build_draft, draft_is_supported

BLOCK = [
    r"\b(?:bypass|disable|ignore|override|skip|evade)\b.{0,45}\b(?:assurance|safety|guardrail|restrictions?|rules?|instructions?)\b",
    r"\b(?:stop|discontinue|double|triple)\b.{0,35}\b(?:my |your )?(?:medication|medicine|dose|treatment|pills?)\b",
    r"\b(?:self[- ]?treat|self[- ]?medicate|without (?:a |my )?(?:doctor|clinician))\b",
    r"\b(?:kill|poison|harm)\b.{0,25}\b(?:myself|someone|patient|person)\b",
]
ESCALATE = [
    r"\b(?:choose|select|recommend|prescribe|pick|decide|start|give|administer|design|create)\b.{0,70}\b(?:chemo\w*|treat\w*|drug|medic\w*|dose|dosage|regimen|therapy|plan)\b",
    r"\b(?:should|can|must)\s+(?:i|we|this patient)\b.{0,60}\b(?:take|use|start|receive|stop|treat|choose)\b",
    r"\b(?:best|right|appropriate)\s+(?:treatment|therapy|drug|medication)\b.{0,30}\b(?:for me|for my|for this patient)\b",
    r"\b(?:diagnose|diagnosis for|stage my|stage this|infer (?:the )?stage|determine (?:the |my )?stage)\b",
    r"\b(?:do i|does (?:this|the) patient|is this|could i)\s+have\b",
    r"\b(?:what|which)\s+(?:cancer\s+)?stage\b",
    r"\b(?:am i|is my|is this)\b.{0,25}\b(?:cancer|malignant|heart attack)\b",
    r"\bwhich\b.{0,40}\b(?:chemo\w*|treat\w*|therapy|drug|medicine)\b.{0,35}\b(?:best|for me|for my|should)\b",
    r"\b(?:tell me whether|determine whether|confirm whether|find out if)\b.{0,60}\b(?:cancer|malignan\w*|heart attack|tumou?r)\b",
    r"\b(?:taper|wean|reduce|increase|change|adjust|switch)\b.{0,40}\b(?:medic\w*|pills?|drug|therapy|dose|treatment)\b",
    r"\bhow\b.{0,20}\b(?:should|can|must)\b.{0,35}\b(?:my|this patient's|the patient's)\b.{0,40}\b(?:treated|managed)\b",
    r"\b(?:recommended|appropriate|suitable|best|right)\b.{0,25}\b(?:treatment|therapy|medicine)\b.{0,25}\b(?:for my|for me|for this patient)\b",
    r"\b(?:treatment|therapy|medicine)\b.{0,25}\b(?:recommended|appropriate|suitable|best|right)\b.{0,25}\b(?:for my|for me|for this patient)\b",
    r"\b(?:tell me|determine|confirm|check|find out)\s+(?:if|whether)\s+(?:i|this patient)\s+(?:have|has)\b",
]
DOSE_TERMS = re.compile(r"\b(?:dose|dosage|dosing|mg/kg|milligrams?|regimen)\b")
DOSE_DEFINITION = re.compile(
    r"^(?:what does (?:a |the )?(?:dose|dosage|dosing) mean|"
    r"what is (?:a |the )?(?:dose|dosage|dosing)|define (?:dose|dosage|dosing))"
    r"(?: in (?:(?:cancer|oncology|cardiology|medical) )?(?:treatment|medicine|care))?[?.]?$"
)
URGENT_SYMPTOMS = [
    r"\bi (?:have|am having|am experiencing|feel|currently have)\b.{0,35}\b(?:chest pain|chest pressure|difficulty breathing|trouble breathing|shortness of breath)\b",
    r"\bmy chest (?:hurts|is hurting|feels tight)\b",
    r"\bi (?:cannot|can't|can not|am struggling to|am unable to)\s+breathe\b",
    r"\bi (?:am|feel)\s+(?:severely )?short of breath\b",
]
URGENT_MESSAGE = (
    "Current chest pain or difficulty breathing needs urgent professional assessment. "
    "Contact your local emergency services now. CareGraph cannot diagnose the cause."
)


def risk_decision(question: str) -> tuple[str | None, str]:
    normalized = " ".join(question.lower().split())
    normalized = normalized.replace("i'm ", "i am ").replace("i’m ", "i am ").replace("can’t", "can't")
    if any(re.search(pattern, normalized) for pattern in URGENT_SYMPTOMS):
        return "escalate", URGENT_MESSAGE
    if any(re.search(pattern, normalized) for pattern in BLOCK):
        return "block", "This request asks for an unsafe action or to bypass the educational safety boundary."
    if any(re.search(pattern, normalized) for pattern in ESCALATE) or (
        DOSE_TERMS.search(normalized) and not DOSE_DEFINITION.fullmatch(normalized)
    ):
        return "escalate", "Diagnosis, staging, medication, dosage, and treatment selection require a qualified clinician."
    return None, ""


def assess_draft(state: dict) -> dict:
    decision, reason = risk_decision(state.get("question", ""))
    if decision:
        final = f"{DISCLAIMER}\n\n{reason}"
        if reason != URGENT_MESSAGE:
            final += " CareGraph cannot make that decision or provide those instructions. Ask a qualified clinician to review the question."
        return {
            "decision": decision,
            "risk_reason": reason,
            "final_answer": final,
            "assurance_checks": ["scope checked", "unsafe draft withheld"],
        }
    if state.get("report_present") and not state.get("report_confirmed"):
        return {
            "decision": "block", "final_answer": "Review and explicitly confirm the extracted synthetic report fields before retrieval.",
            "assurance_checks": ["unconfirmed report withheld"],
        }
    draft = state.get("draft_answer", "")
    if draft_is_supported(draft, state):
        return {"decision": "release", "final_answer": draft, "assurance_checks": ["scope checked", "every excerpt and citation matched", "no added claims"]}
    rewritten = build_draft(state)
    if draft_is_supported(rewritten, state):
        return {"decision": "rewrite", "final_answer": rewritten, "assurance_checks": ["unsupported draft removed", "rewritten from exact evidence", "rewritten output revalidated"]}
    return {"decision": "block", "final_answer": f"{DISCLAIMER}\n\nThe draft could not be verified and was withheld.", "assurance_checks": ["verification failed"]}
