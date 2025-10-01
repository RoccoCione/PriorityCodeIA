# src/features.py

def preprocess_input(data: dict) -> dict:
    """
    Converte i valori grezzi del form (vitali + sintomi)
    in categorie utili per il motore a regole e, in futuro, per NB.
    """
    processed = {}

    # --- Vitali ---
    # SpO2 (saturazione ossigeno, %)
    spo2 = data.get("SpO2", None)
    if spo2 is None:
        processed["spo2_cat"] = "unknown"
    elif spo2 <= 90:
        processed["spo2_cat"] = "severa"
    elif spo2 <= 94:
        processed["spo2_cat"] = "moderata"
    else:
        processed["spo2_cat"] = "ok"

    # SBP (pressione sistolica, mmHg)
    sbp = data.get("SBP", None)
    if sbp is None:
        processed["sbp_cat"] = "unknown"
    elif sbp < 90:
        processed["sbp_cat"] = "severa"
    else:
        processed["sbp_cat"] = "ok"

    # RR (atti/min)
    rr = data.get("RR", None)
    if rr is None:
        processed["rr_cat"] = "unknown"
    elif rr >= 30:
        processed["rr_cat"] = "alta"
    else:
        processed["rr_cat"] = "ok"

    # Temp (°C)
    temp = data.get("Temp", None)
    if temp is None:
        processed["temp_cat"] = "unknown"
    elif temp >= 39.0:
        processed["temp_cat"] = "alta"
    else:
        processed["temp_cat"] = "ok"

    # --- Sintomi binari (0/1/None) -> no/yes/unknown ---
    def to_yn(val):
        if val is None:
            return "unknown"
        return "yes" if int(val) == 1 else "no"

    for s in [
        "dolore_toracico",
        "dispnea",
        "alterazione_coscienza",
        "trauma_magg",
        "sanguinamento_massivo",
    ]:
        processed[s] = to_yn(data.get(s, None))

    return processed
