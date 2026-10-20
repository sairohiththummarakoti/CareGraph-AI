"""Bounded report parsing. Uploaded bytes and extracted values are never saved."""

import io
import re
from pathlib import Path

MAX_REPORT_BYTES = 2 * 1024 * 1024
MAX_REPORT_PAGES = 20
MAX_REPORT_TEXT = 60_000
FACT_FIELDS = ("tumor_site", "histology", "grade", "stage", "tumor_size", "biomarkers")
ALIASES = {
    "tumor site": "tumor_site", "tumour site": "tumor_site", "site": "tumor_site",
    "histology": "histology", "histologic type": "histology",
    "grade": "grade", "tumor grade": "grade",
    "stage": "stage", "pathologic stage": "stage",
    "tumor size": "tumor_size", "tumour size": "tumor_size", "size": "tumor_size",
    "biomarkers": "biomarkers", "biomarker": "biomarkers", "biomarker status": "biomarkers",
}
BIOMARKER_LABELS = {"er": "ER", "pr": "PR", "her2": "HER2", "pd-l1": "PD-L1", "ki-67": "Ki-67"}
UNKNOWN = re.compile(
    r"^(?:unknown|not provided|not reported|not available|not assessed|not tested|"
    r"not specified|pending|n/?a|none|unspecified|missing|undetermined|tbd|[-?]+)"
    r"(?:$|(?=[\s.!;,:()\[\]-]).*)", re.I
)
ASSAY_UNKNOWN = re.compile(
    r"\b(?:er|pr|her2|pd-l1|ki-67)\b\s*(?:(?:status|result)\s*)?[:=-]?\s*"
    r"(?:unknown|not provided|not reported|not available|not assessed|not tested|pending|"
    r"unspecified|missing|undetermined|tbd)\b", re.I
)


class ReportError(ValueError):
    """An input cannot be used as a bounded, explicitly synthetic report."""


def _report_text(data: bytes, name: str) -> str:
    if not isinstance(data, bytes) or not data:
        raise ReportError("The report is empty. Choose a synthetic TXT or text-based PDF.")
    if len(data) > MAX_REPORT_BYTES:
        raise ReportError("The report exceeds the 2 MiB limit.")
    extension = Path(name).suffix.lower()
    if extension == ".txt":
        try:
            text = data.decode("utf-8-sig", errors="strict")
        except UnicodeDecodeError as exc:
            raise ReportError("TXT reports must use UTF-8 encoding.") from exc
    elif extension == ".pdf":
        if not data.startswith(b"%PDF-"):
            raise ReportError("This is not a valid PDF file.")
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(data), strict=True)
            if reader.is_encrypted:
                raise ReportError("Encrypted PDFs are not supported. Use a synthetic unencrypted report.")
            if len(reader.pages) > MAX_REPORT_PAGES:
                raise ReportError("PDF reports are limited to 20 pages.")
            extracted = []
            for page in reader.pages:
                extracted.append(page.extract_text() or "")
                if sum(map(len, extracted)) > MAX_REPORT_TEXT:
                    raise ReportError("The report contains too much extracted text.")
            text = "\n".join(extracted)
        except ReportError:
            raise
        except Exception as exc:
            raise ReportError("The PDF could not be parsed. Use a valid text-based synthetic PDF.") from exc
        if not text.strip():
            raise ReportError("This PDF has no readable text. Scanned/image-only PDFs are not supported.")
    else:
        raise ReportError("Unsupported report type. Use a synthetic .txt or .pdf file.")
    if not text.strip():
        raise ReportError("The report contains no usable text.")
    if "\x00" in text or len(text) > MAX_REPORT_TEXT:
        raise ReportError("The report contains invalid or excessive text.")
    if not re.search(r"(?im)^\s*SYNTHETIC(?:\s+PATHOLOGY)?\s+REPORT\b", text[:2048]):
        raise ReportError("The report must begin with an explicit SYNTHETIC REPORT or SYNTHETIC PATHOLOGY REPORT label.")
    return text


def parse_report(data: bytes, name: str) -> dict:
    """Extract explicit labeled fields only; absent, blank or unknown values stay missing."""
    text = _report_text(data, name)
    facts: dict[str, str] = {}
    seen: set[str] = set()
    biomarkers: dict[str, str] = {}
    unknown_markers: list[str] = []
    empty_field: str | None = None
    for line in text.splitlines():
        if not line.strip():
            continue
        match = re.match(r"^\s*([A-Za-z0-9 _-]+)\s*:\s*(.*?)\s*$", line)
        if not match:
            if empty_field:
                raise ReportError(
                    f"The {empty_field} field has a wrapped continuation. Put its value on the same line after the colon, or use Unknown."
                )
            continue
        empty_field = None
        label, value = match.groups()
        marker = BIOMARKER_LABELS.get(label.lower().strip())
        if marker:
            if marker in seen:
                raise ReportError(f"Duplicate {marker} fields are ambiguous. Keep one explicit value.")
            seen.add(marker)
            if len(value) > 240:
                raise ReportError(f"The {marker} field exceeds the 240-character limit.")
            empty_field = marker if not value else None
            if value and not UNKNOWN.fullmatch(value):
                biomarkers[marker] = value
            else:
                unknown_markers.append("biomarker_" + marker.lower())
            continue
        key = ALIASES.get(re.sub(r"[_-]+", " ", label.lower()).strip())
        if key is None:
            continue
        if key in seen:
            raise ReportError(f"Duplicate {key.replace('_', ' ')} fields are ambiguous. Keep one explicit value.")
        seen.add(key)
        if len(value) > 240:
            raise ReportError(f"The {key.replace('_', ' ')} field exceeds the 240-character limit.")
        empty_field = key.replace("_", " ") if not value else None
        if key == "biomarkers" and ASSAY_UNKNOWN.search(value):
            raise ReportError("The Biomarkers field contains an unknown assay. Use individual fields such as ER: and HER2: so unknown results remain explicit.")
        if value and not UNKNOWN.fullmatch(value):
            facts[key] = value
    if not seen:
        raise ReportError("No recognized tumor fields were found. Use labeled fields such as Tumor site: and Histology:.")
    if "biomarkers" in seen and any(marker in seen for marker in BIOMARKER_LABELS.values()):
        raise ReportError("Use either a Biomarkers field or individual biomarker fields to avoid ambiguous transcription.")
    if biomarkers:
        combined = "; ".join(f"{marker}: {value}" for marker, value in biomarkers.items())
        if len(combined) > 240:
            raise ReportError("The combined biomarkers field exceeds the 240-character limit. Shorten the individual values.")
        facts["biomarkers"] = combined
    return {"pathology_facts": facts, "missing_facts": [field for field in FACT_FIELDS if field not in facts] + unknown_markers}
