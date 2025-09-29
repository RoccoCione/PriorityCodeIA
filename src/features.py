def preprocess_input(data: dict) -> dict:
    """
    Converte i valori grezzi dei sintomi/vitali
    in categorie utili per le regole e per Naive Bayes.
    """
    processed = {}

    # Saturazione ossigeno
    spo2 = data.get("SpO2", None)
    if spo2 is None:
        processed["spo2_cat"] = "unknown"
    elif spo2 <= 90:
        processed["spo2_cat"] = "severa"
    elif spo2 <= 94:
        processed["spo2_cat"] = "moderata"
    else:
        processed["spo2_cat"] = "ok"

    # Pressione sistolica (SBP)
    sbp = data.get("SBP", None)
    if sbp is None:
        processed["sbp_cat"] = "unknown"
    elif sbp < 90:
        processed["sbp_cat"] = "severa"
    else:
        processed["sbp_cat"] = "ok"

    # Sintomi binari (0/1/None)
    for s in ["dolore_toracico", "dispnea", "alterazione_coscienza"]:
        val = data.get(s, None)
        if val is None:
            processed[s] = "unknown"
        elif val == 1:
            processed[s] = "yes"
        else:
            processed[s] = "no"

    return processed
