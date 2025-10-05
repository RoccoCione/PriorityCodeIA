# src/features.py
"""
Discretizzazione dei dati grezzi in feature categoriali coerenti col FEATURE_SPACE:
- spo2_cat: severa (<90), moderata (90–93), ok (>=94), unknown
- sbp_cat:  severa (<90), ok (>=90), unknown
- rr_cat:   alta (>=24), ok (<24), unknown
- temp_cat: alta (>=38.0), ok (<38.0), unknown
Sintomi tri-stato: 1->"yes", 0->"no", None->"unknown"
"""

from typing import Optional, Dict

# ---- soglie cliniche (semplificate) ----
SPO2_SEVERE_LT = 90       # < 90 -> severa
SPO2_MOD_LT    = 94       # [90..93] -> moderata, >=94 -> ok
SBP_SEVERE_LT  = 90       # < 90 -> severa
RR_HIGH_GE     = 24       # >= 24 -> alta
TEMP_HIGH_GE   = 38.0     # >= 38.0 -> alta

# --------- mapping funzioni ---------

def cat_spo2(spo2):
    if spo2 is None: return "unknown"
    return "severa" if spo2 < 90 else ("moderata" if spo2 < 94 else "ok")

def cat_sbp(sbp):
    if sbp is None: return "unknown"
    return "severa" if sbp < 90 else "ok"

def cat_rr(rr):
    if rr is None: return "unknown"
    return "alta" if rr >= 24 else "ok"

def cat_temp(t):
    if t is None: return "unknown"
    return "alta" if t >= 38.0 else "ok"

def tri_to_str(v):
    return "yes" if v == 1 else ("no" if v == 0 else "unknown")


# --------- preprocess principale ---------
def preprocess_input(raw: Dict) -> Dict[str, str]:
    """
    Converte l'input grezzo (numeri + tri-stato 1/0/None) in feature categoriali.
    Restituisce un dizionario allineato allo schema atteso da rules_engine / naive_bayes.
    """
    facts = {
        "spo2_cat": cat_spo2(raw.get("SpO2")),
        "sbp_cat":  cat_sbp(raw.get("SBP")),
        "rr_cat":   cat_rr(raw.get("RR")),
        "temp_cat": cat_temp(raw.get("Temp")),
        "dolore_toracico":       tri_to_str(raw.get("dolore_toracico")),
        "dispnea":               tri_to_str(raw.get("dispnea")),
        "alterazione_coscienza": tri_to_str(raw.get("alterazione_coscienza")),
        "trauma_magg":           tri_to_str(raw.get("trauma_magg")),
        "sanguinamento_massivo": tri_to_str(raw.get("sanguinamento_massivo")),
    }
    return facts
