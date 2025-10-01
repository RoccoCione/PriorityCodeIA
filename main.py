# main.py
from typing import Optional
from src.features import preprocess_input
from src.rules_engine import forward_chain, explain_rules
from src.priority_queue import TriageQueue

def ask_float(prompt: str) -> Optional[float]:
    while True:
        s = input(prompt).strip()
        if s == "":
            return None
        try:
            return float(s.replace(",", "."))
        except ValueError:
            print("Valore non valido. Inserisci un numero (oppure premi Invio per saltare).")

def ask_int(prompt: str) -> Optional[int]:
    while True:
        s = input(prompt).strip()
        if s == "":
            return None
        try:
            return int(s)
        except ValueError:
            print("Valore non valido. Inserisci un intero (oppure premi Invio per saltare).")

def ask_yn_unknown(prompt: str) -> Optional[int]:
    while True:
        s = input(prompt + " [s=si / n=no / Invio=incerto]: ").strip().lower()
        if s == "":
            return None
        if s in ("s", "si", "y", "yes"):
            return 1
        if s in ("n", "no"):
            return 0
        print("Risposta non valida. Digita 's', 'n' oppure premi Invio per incerto.")

def acquire_patient_input() -> dict:
    print("\n--- Inserimento nuovi dati paziente ---")
    name = input("Nome/etichetta paziente (es. 'Mario R.'): ").strip() or "Anonimo"
    spo2 = ask_float("SpO2 (%) [es. 97, invio per non disponibile]: ")
    sbp  = ask_int("Pressione sistolica SBP (mmHg) [es. 120, invio per ND]: ")
    rr   = ask_int("Frequenza respiratoria RR (atti/min) [es. 18, invio per ND]: ")
    temp = ask_float("Temperatura (°C) [es. 37.2, invio per ND]: ")
    dolore_toracico      = ask_yn_unknown("Dolore toracico?")
    dispnea              = ask_yn_unknown("Dispnea (fiato corto)?")
    alterazione_coscienza = ask_yn_unknown("Alterazione dello stato di coscienza?")
    trauma_magg          = ask_yn_unknown("Trauma maggiore?")
    sanguinamento_mass   = ask_yn_unknown("Sanguinamento massivo?")

    return {
        "name": name,
        "SpO2": spo2, "SBP": sbp, "RR": rr, "Temp": temp,
        "dolore_toracico": dolore_toracico,
        "dispnea": dispnea,
        "alterazione_coscienza": alterazione_coscienza,
        "trauma_magg": trauma_magg,
        "sanguinamento_massivo": sanguinamento_mass,
    }

def triage_and_enqueue(queue: TriageQueue, raw_input: dict):
    facts = preprocess_input(raw_input)
    triage, fired, facts_out = forward_chain(facts)
    patient = queue.enqueue(triage=triage, payload={"name": raw_input.get("name","Anonimo"),
                                                    "facts": facts_out, "rules": fired})
    pos = queue.get_position(patient.patient_id)

    print("\n" + "=" * 78)
    print(f"[ARRIVO] {raw_input.get('name','Anonimo')} -> triage={triage}  id={patient.patient_id}  posizione={pos}")
    print("Regole attivate:")
    print(explain_rules(fired))
    print("\nStato coda (globale):")
    print(queue.pretty_print())
    print()

def main():
    print("=== Simulatore Triage (CLI) ===")
    print("Comandi: [n] nuovo paziente, [s] stato coda, [r] rimuovi per ID, [x] servi prossimo, [q] esci.\n")

    q = TriageQueue()

    while True:
        cmd = input("Comando (n/s/r/x/q): ").strip().lower()
        if cmd == "q":
            print("Uscita. A presto.")
            break

        elif cmd == "s":
            print("\n-- Stato coda --")
            print(q.pretty_print(), "\n")

        elif cmd == "n":
            raw = acquire_patient_input()
            triage_and_enqueue(q, raw)

        elif cmd == "r":
            pid = input("ID paziente da rimuovere (es. P0003): ").strip()
            if not pid:
                print("Nessun ID inserito.\n")
                continue
            removed = q.remove(pid)
            if removed:
                print(f"RIMOSSO: {removed.patient_id} | {removed.triage} | name={removed.payload.get('name')}")
            else:
                print("ID non trovato.")
            print("\n-- Stato coda --")
            print(q.pretty_print(), "\n")

        elif cmd == "x":
            served = q.serve_next()
            if served:
                print(f"SERVITO (rimosso dalla coda): {served.patient_id} | {served.triage} | name={served.payload.get('name')}")
            else:
                print("Coda vuota, nessuno da servire.")
            print("\n-- Stato coda --")
            print(q.pretty_print(), "\n")

        else:
            print("Comando non riconosciuto. Usa n/s/r/x/q.\n")

if __name__ == "__main__":
    main()
