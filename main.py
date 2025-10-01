# main.py
"""
CLI definitiva (commentata) per il tuo Triage:
----------------------------------------------
Flusso per ogni paziente:
1) Preprocess (features.py) -> facts categoriali.
2) Regole (rules_engine.forward_chain):
   - se 'Rosso' => override duro (stop).
   - altrimenti calcola 'triage_min' (vincoli tipo "almeno Giallo").
3) Naive Bayes (src/naive_bayes):
   - stima P(y|x) e sceglie la classe che MINIMIZZA il COSTO ATTESO (prudente).
   - rispetta 'triage_min' (non scendere sotto).
4) Inserimento nella coda a priorità + stampa posizione e spiegazioni.

Comandi:
  n = nuovo paziente
  s = stato coda
  r = rimuovi per ID
  x = servi prossimo
  q = esci
"""

from typing import Optional

from src.features import preprocess_input
from src.rules_engine import forward_chain, explain_rules, default_kb
from src.priority_queue import TriageQueue
from src.naive_bayes import ensure_trained, NBModel, max_severity

# ----------- helper: ricava 'triage_min' dalle regole attivate -----------
def compute_triage_min_from_fired(fired_ids):
    """
    Cerca tra le regole attivate se ci sono 'then' con triage_min,
    e ne prende il massimo (cioè il livello più severo).
    """
    kb = {r.id: r for r in default_kb()}
    tri_min = None
    for rid in fired_ids:
        r = kb.get(rid)
        if r and "triage_min" in r.then:
            tri_min = r.then["triage_min"] if tri_min is None else max_severity(tri_min, r.then["triage_min"])
    return tri_min

# ----------- fusione decisionale: Regole + NB cost-sensitive -----------
def triage_with_rules_and_nb(raw: dict, nb_model: NBModel):
    """
    1) Preprocess -> facts categoriali
    2) Regole -> se Rosso => stop (override)
    3) triage_min dalle regole, se presente
    4) NB -> decisione prudente rispettando triage_min
    """
    # 1) Preprocess: numeri/si-no-incerto -> categorie ('ok','severa','yes','unknown',...)
    facts = preprocess_input(raw)

    # 2) Regole simboliche (forward chaining)
    triage_rules, fired, facts_out = forward_chain(facts)

    # Override: se regola critica ha dato ROSSO, non consultiamo NB
    if triage_rules == "Rosso":
        probs = {"Bianco": 0.0, "Verde": 0.0, "Giallo": 0.0, "Rosso": 1.0}
        return triage_rules, fired, facts_out, probs

    # 3) Calcolo del minimo garantito (es. "almeno Giallo")
    triage_min = compute_triage_min_from_fired(fired)

    # 4) Decisione NB cost-sensitive (prudente) + rispetto del triage_min
    yhat, probs = nb_model.decide_cost_sensitive(facts_out, triage_min=triage_min)
    return yhat, fired, facts_out, probs

# --------------------------- IO helpers (CLI) ---------------------------
def ask_float(prompt: str) -> Optional[float]:
    s = input(prompt).strip().replace(",", ".")
    if s == "": return None
    try: return float(s)
    except ValueError:
        print("Valore non valido. (Invio = ND)"); return ask_float(prompt)

def ask_int(prompt: str) -> Optional[int]:
    s = input(prompt).strip()
    if s == "": return None
    try: return int(s)
    except ValueError:
        print("Valore non valido. (Invio = ND)"); return ask_int(prompt)

def ask_tri(prompt: str) -> Optional[int]:
    """
    Sì/No/Incerto -> 1/0/None
    """
    s = input(prompt + " [s=si / n=no / Invio=incerto]: ").strip().lower()
    if s == "": return None
    if s in ("s", "si", "y", "yes"): return 1
    if s in ("n", "no"): return 0
    print("Risposta non valida."); return ask_tri(prompt)

# --------------------------- MAIN LOOP ---------------------------
def main():
    print("=== Triage con Regole + Naive Bayes (CSV reale) ===")
    print("Userò 'data/examples.csv' per allenare NB se non trovo 'data/nb_model.pkl'.\n")

    # 1) Carica/Allena NB dal tuo CSV categoriale (nessuna generazione automatica)
    nb_model = ensure_trained(
        model_path="data/nb_model.pkl",
        csv_path="data/examples.csv"   # <--- il tuo dataset statico
    )

    # 2) Istanzia la coda a priorità
    q = TriageQueue()

    # 3) Loop comandi
    while True:
        cmd = input("\nComando (n=nuovo, s=stato, r=rimuovi ID, x=servi prossimo, q=esci): ").strip().lower()

        if cmd == "q":
            print("Uscita. A presto.")
            break

        elif cmd == "s":
            print("\n-- Stato coda --")
            print(q.pretty_print())

        elif cmd == "r":
            pid = input("ID paziente da rimuovere (es. P0003): ").strip()
            if not pid:
                print("Nessun ID inserito."); continue
            removed = q.remove(pid)
            if removed:
                print(f"RIMOSSO: {removed.patient_id} | {removed.triage} | name={removed.payload.get('name','')}")
            else:
                print("ID non trovato.")
            print("\n-- Stato coda --")
            print(q.pretty_print())

        elif cmd == "x":
            served = q.serve_next()
            if served:
                print(f"SERVITO: {served.patient_id} | {served.triage} | name={served.payload.get('name','')}")
            else:
                print("Coda vuota.")
            print("\n-- Stato coda --")
            print(q.pretty_print())

        elif cmd == "n":
            # --- Raccolta input semplice (accetta Invio=ND/Incerto) ---
            name = input("Nome/etichetta (opzionale): ").strip() or "Anonimo"
            spo2 = ask_int("SpO2 (%) [Invio=ND]: ")
            sbp  = ask_int("SBP (mmHg) [Invio=ND]: ")
            rr   = ask_int("RR (atti/min) [Invio=ND]: ")
            temp = ask_float("Temperatura (°C) [Invio=ND]: ")

            print("Sintomi:")
            dolore = ask_tri("  Dolore toracico?")
            disp   = ask_tri("  Dispnea?")
            cosc   = ask_tri("  Alterazione coscienza?")
            trauma = ask_tri("  Trauma maggiore?")
            sang   = ask_tri("  Sanguinamento massivo?")

            raw = {
                "name": name,
                "SpO2": spo2, "SBP": sbp, "RR": rr, "Temp": temp,
                "dolore_toracico": dolore, "dispnea": disp,
                "alterazione_coscienza": cosc, "trauma_magg": trauma, "sanguinamento_massivo": sang,
            }

            # --- FUSIONE: Regole -> (Rosso? stop) / altrimenti NB prudente ---
            label, fired, facts_out, probs = triage_with_rules_and_nb(raw, nb_model)

            # --- Enqueue + stampa risultato + spiegazioni ---
            patient = q.enqueue(label, payload={"name": name, "facts": facts_out, "rules": fired, "probs": probs})
            pos = q.get_position(patient.patient_id)

            print("\n--- ESITO ---")
            print(f"Codice: {label} | Posizione: {pos} su {len(q)}")
            print("Probabilità NB:", {k: round(v, 3) for k, v in probs.items()})
            print("Regole attivate:\n" + explain_rules(fired))
            print("\n-- Coda --")
            print(q.pretty_print())

        else:
            print("Comando non riconosciuto. Usa n/s/r/x/q.")

if __name__ == "__main__":
    main()
