# src/rules_engine.py
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional

Condition = Tuple[str, Any]  # es. ("spo2_cat", "severa")

@dataclass
class Rule:
    id: str
    any_conditions: List[Condition] = field(default_factory=list)  # almeno una vera
    all_conditions: List[Condition] = field(default_factory=list)  # tutte vere
    then: Dict[str, Any] = field(default_factory=dict)             # es. {"triage": "Rosso"}
    priority: int = 0                                              # più alto -> più urgente

    @property
    def specificity(self) -> int:
        return len(self.any_conditions) + len(self.all_conditions)

    def matches(self, facts: Dict[str, Any]) -> bool:
        # ALL
        for k, v in self.all_conditions:
            if facts.get(k) != v:
                return False
        # ANY (se presente almeno una dev'essere vera)
        if self.any_conditions:
            return any(facts.get(k) == v for k, v in self.any_conditions)
        return True  # nessuna ANY -> ok

# ---------- Knowledge Base ----------
def default_kb() -> List[Rule]:
    return [
        # CRITICHE → override Rosso
        Rule(
            id="R1_ipossia_shock_coscienza",
            any_conditions=[("spo2_cat", "severa"), ("sbp_cat", "severa"), ("alterazione_coscienza", "yes")],
            then={"triage": "Rosso"},
            priority=100,
        ),
        Rule(
            id="R2_trauma_o_sanguinamento",
            any_conditions=[("trauma_magg", "yes"), ("sanguinamento_massivo", "yes")],
            then={"triage": "Rosso"},
            priority=95,
        ),

        # IMPORTANTI → almeno Giallo
        Rule(
            id="R3_dolore_toracico_e_dispnea",
            all_conditions=[("dolore_toracico", "yes"), ("dispnea", "yes")],
            then={"triage_min": "Giallo"},
            priority=70,
        ),
        Rule(
            id="R4_tachipnea_moderata_con_dispnea",
            all_conditions=[("rr_cat", "alta"), ("dispnea", "yes")],
            then={"triage_min": "Giallo"},
            priority=60,
        ),

        # NON critiche → Verde (se nessun allarme)
        Rule(
            id="R5_febbre_alta_senza_allarme",
            all_conditions=[("temp_cat", "alta"), ("alterazione_coscienza", "no"), ("dispnea", "no")],
            then={"triage": "Verde"},
            priority=40,
        ),

        # FALLBACK
        Rule(
            id="R6_fallback_bianco",
            then={"triage": "Bianco"},
            priority=0,
        ),
    ]

ORDER = ["Bianco", "Verde", "Giallo", "Rosso"]

def max_severity(a: str, b: str) -> str:
    return a if ORDER.index(a) >= ORDER.index(b) else b

def forward_chain(facts: Dict[str, Any], kb: Optional[List[Rule]] = None) -> tuple[str, List[str], Dict[str, Any]]:
    """
    Ritorna: (triage_finale, regole_attivate, facts_finali)
    Risoluzione conflitti: priority DESC, specificity DESC, ordine in lista.
    Supporta 'triage' (hard set) e 'triage_min' (minimo garantito).
    """
    if kb is None:
        kb = default_kb()

    fired: List[str] = []
    facts = dict(facts)
    changed = True
    triage_min: Optional[str] = None

    while changed:
        changed = False
        candidates = [r for r in kb if r.id not in fired and r.matches(facts)]
        if not candidates:
            break
        candidates.sort(key=lambda r: (r.priority, r.specificity), reverse=True)
        r = candidates[0]

        # applica effetti
        if "triage" in r.then:
            facts["triage"] = r.then["triage"]
        if "triage_min" in r.then:
            triage_min = r.then["triage_min"] if triage_min is None else max_severity(triage_min, r.then["triage_min"])

        fired.append(r.id)
        changed = True

        # se già Rosso → stop
        if facts.get("triage") == "Rosso":
            break

    # consolidamento minimo garantito
    triage = facts.get("triage", "Bianco")
    if triage_min is not None:
        triage = max_severity(triage, triage_min)
        facts["triage"] = triage

    return triage, fired, facts

def explain_rules(fired_ids: List[str], kb: Optional[List[Rule]] = None) -> str:
    kb = kb or default_kb()
    by_id = {r.id: r for r in kb}
    lines = []
    for rid in fired_ids:
        r = by_id[rid]
        cond_any = " OR ".join([f"{k}={v}" for k, v in r.any_conditions]) if r.any_conditions else ""
        cond_all = " AND ".join([f"{k}={v}" for k, v in r.all_conditions]) if r.all_conditions else ""
        if cond_any and cond_all:
            cond = f"({cond_all}) OR ({cond_any})"
        else:
            cond = cond_all or cond_any or "TRUE"
        then_str = ", ".join([f"{k}→{v}" for k, v in r.then.items()])
        lines.append(f"- {rid}: se {cond} allora {then_str} [prio={r.priority}]")
    return "\n".join(lines)
