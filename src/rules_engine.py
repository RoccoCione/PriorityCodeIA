# src/rules_engine.py
"""
Motore a regole (forward chaining semplificato) con priorità e spiegazioni.
- Regole CRITICHE: assegnano direttamente Rosso (override).
- Regole di SUPPORTO: impongono un 'triage_min' (almeno Giallo/Verde).
Nessuna regola critica usa 'unknown' -> evita falsi positivi Rosso.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ordine di severità (da meno a più grave)
ORDER = ["Bianco", "Verde", "Giallo", "Rosso"]

def max_severity(a: str, b: str) -> str:
    return a if ORDER.index(a) >= ORDER.index(b) else b

@dataclass
class Rule:
    id: str
    if_: Dict[str, str]          # condizioni esatte key==value (AND)
    then: Dict[str, str]         # {"triage": "..."} oppure {"triage_min": "..."}
    priority: int                # per risoluzione conflitti: numeri più alti = più importanti
    desc: str                    # descrizione leggibile

def default_kb() -> List[Rule]:
    """
    Conoscenza clinica (semplificata) con priorità:
    100 = critiche (Rosso),
     50 = minimi garantiti (>= Giallo),
     20 = minimi leggeri (>= Verde)
    """
    rules: List[Rule] = []

    # ---- CRITICHE -> ROSSO (override) ----
    rules += [
        Rule("R_CRIT_SPO2", {"spo2_cat": "severa"}, {"triage": "Rosso"}, 100,
             "SpO₂ severamente bassa"),
        Rule("R_CRIT_SBP", {"sbp_cat": "severa"}, {"triage": "Rosso"}, 100,
             "Ipotensione severa (shock)"),
        Rule("R_CRIT_COSC", {"alterazione_coscienza": "yes"}, {"triage": "Rosso"}, 100,
             "Alterazione dello stato di coscienza"),
        Rule("R_CRIT_TRAUMA", {"trauma_magg": "yes"}, {"triage": "Rosso"}, 100,
             "Trauma maggiore"),
        Rule("R_CRIT_BLEED", {"sanguinamento_massivo": "yes"}, {"triage": "Rosso"}, 100,
             "Sanguinamento massivo"),
    ]

    # ---- MINIMI GARANTITI (>= Giallo) ----
    rules += [
        Rule("R_MIN_TORACE_DISP", {"dolore_toracico": "yes", "dispnea": "yes"},
             {"triage_min": "Giallo"}, 50, "Dolore toracico + Dispnea ⇒ almeno Giallo"),
        Rule("R_MIN_RR_DISP", {"rr_cat": "alta", "dispnea": "yes"},
             {"triage_min": "Giallo"}, 50, "Tachipnea + Dispnea ⇒ almeno Giallo"),
    ]

    # ---- MINIMI LEGGERI (>= Verde) ----
    rules += [
        Rule("R_MIN_FEBBRE_SENZA_DISP", {"temp_cat": "alta", "dispnea": "no", "alterazione_coscienza": "no"},
             {"triage_min": "Verde"}, 20, "Febbre alta senza dispnea/alterazione ⇒ almeno Verde"),
    ]

    return rules

def _match(rule: Rule, facts: Dict[str, str]) -> bool:
    # tutte le condizioni devono essere rispettate esattamente
    for k, v in rule.if_.items():
        if facts.get(k) != v:
            return False
    return True

def forward_chain(facts: Dict[str, str]) -> Tuple[str, List[str], Dict[str, str]]:
    """
    Esegue il matching di tutte le regole e risolve i conflitti.
    Ritorna:
      - triage_rules: colore assegnato dalle regole (Rosso se critiche, altrimenti Bianco)
      - fired: lista ID regole attivate (per spiegazioni)
      - facts_out: copia dei facts (immutati)
    Nota: il 'triage_min' sarà letto fuori (es. da Streamlit) usando default_kb().
    """
    kb = default_kb()
    fired: List[Rule] = []

    # ordina per priorità decrescente (critiche prima), a parità mantieni ordine dichiarazione
    kb_sorted = sorted(kb, key=lambda r: r.priority, reverse=True)

    # colleziona tutte le regole che matchano
    for r in kb_sorted:
        if _match(r, facts):
            fired.append(r)

    # se qualunque regola assegna triage diretto, scegli il più severo tra quelle
    triage_direct = None
    for r in fired:
        if "triage" in r.then:
            tri = r.then["triage"]
            triage_direct = tri if triage_direct is None else max_severity(triage_direct, tri)

    if triage_direct is not None:
        # override (es. Rosso)
        return triage_direct, [r.id for r in fired], dict(facts)

    # nessuna diretta -> le regole attivate (se presenti) impongono minimi (letto a valle)
    # per compatibilità, ritorniamo "Bianco" come triage delle sole regole
    return "Bianco", [r.id for r in fired], dict(facts)

def explain_rules(fired_ids: List[str]) -> str:
    """Restituisce testo leggibile con ID e descrizione delle regole attivate."""
    kb = {r.id: r for r in default_kb()}
    lines = []
    for rid in fired_ids:
        r = kb.get(rid)
        if r:
            action = r.then.get("triage") or f"≥ {r.then.get('triage_min')}"
            lines.append(f"- [{rid}] (prio {r.priority}) → {action}: {r.desc}")
    return "\n".join(lines) if lines else "—"
