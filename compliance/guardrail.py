"""
FasalSetu Compliance Guardrail
Audit log is written to logs/compliance_audit.jsonl so entries persist across reloads.
"""
import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("guardrail")

_BASE_DIR   = Path(__file__).parent.parent
_LOG_DIR    = _BASE_DIR / "logs"
_AUDIT_FILE = _LOG_DIR / "compliance_audit.jsonl"
_LOG_DIR.mkdir(exist_ok=True)

_BANNED_PATH = Path(__file__).parent / "banned_pesticides.json"
try:
    with open(_BANNED_PATH) as f:
        _BANNED = json.load(f)
except FileNotFoundError:
    logger.warning("banned_pesticides.json not found — using built-in list")
    _BANNED = {
        "banned": [
            "Endosulfan", "Monocrotophos", "Methyl Parathion", "Phosphamidon",
            "Triazophos", "Chlorpyrifos", "Dichlorvos", "Aluminium Phosphide",
            "Methomyl", "Carbofuran", "Aldrin", "Dieldrin", "DDT",
        ],
        "restricted": {
            "Glyphosate":   "Not permitted on food crops without state approval",
            "Atrazine":     "Restricted to maize and sugarcane only",
            "2,4-D":        "Do not apply within 100m of water bodies",
            "Cypermethrin": "Do not apply during flowering — toxic to pollinators",
        },
        "license_required": ["Methyl Bromide", "Aluminium Phosphide"],
    }


class ComplianceViolation(Exception):
    pass


def _write_audit_entry(entry: dict) -> None:
    try:
        with open(_AUDIT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error("Failed to write audit entry: %s", e)


def get_audit_log(last_n: int = 50) -> list:
    if not _AUDIT_FILE.exists():
        return []
    try:
        lines = _AUDIT_FILE.read_text(encoding="utf-8").strip().splitlines()
        entries = []
        for line in lines:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return entries[-last_n:]
    except Exception as e:
        logger.error("Failed to read audit log: %s", e)
        return []


def check_and_gate(agent_name: str, response: dict) -> dict:
    violations: list[str] = []
    warnings:   list[str] = []
    text_lower = json.dumps(response, ensure_ascii=False).lower()

    for substance in _BANNED.get("banned", []):
        if substance.lower() in text_lower:
            violations.append(
                f"BANNED: '{substance}' is banned under the Insecticides Act 1968."
            )

    for substance, restriction in _BANNED.get("restricted", {}).items():
        if substance.lower() in text_lower:
            warnings.append(f"RESTRICTED: '{substance}' — {restriction}")

    for substance in _BANNED.get("license_required", []):
        if substance.lower() in text_lower:
            warnings.append(f"LICENSE REQUIRED: '{substance}' requires a certified operator licence.")

    entry = {
        "timestamp":  datetime.now().isoformat(timespec="seconds"),
        "agent":      agent_name,
        "violations": violations,
        "warnings":   warnings,
        "blocked":    len(violations) > 0,
    }
    _write_audit_entry(entry)
    logger.info("AUDIT | agent=%s violations=%d warnings=%d", agent_name, len(violations), len(warnings))

    if violations:
        raise ComplianceViolation("Response blocked:\n" + "\n".join(violations))

    if warnings:
        response["_compliance_warnings"] = warnings

    return response
