# main.py
from src.features import preprocess_input
from src.rules_engine import forward_chain, explain_rules

def run_case(name: str, sample: dict):
    facts = preprocess_input(sample)
    triage, fired, facts_out = forward_chain(facts)
    print("="*70)
    print(f"[CASO] {name}")
    print("Input grezzi:", sample)
    print("Facts:", facts)
    print("Triage:", triage)
    print("Regole attivate:")
    print(explain_rules(fired))
    print("Facts finali:", facts_out)

if __name__ == "__main__":
    # Caso 1: ipossia grave → Rosso
    case1 = {
        "SpO2": 88, "SBP": 120, "RR": 18, "Temp": 37.2,
        "dolore_toracico": 0, "dispnea": 0, "alterazione_coscienza": 0,
        "trauma_magg": 0, "sanguinamento_massivo": 0
    }

    # Caso 2: dolore toracico + dispnea, vitali non critici → almeno Giallo
    case2 = {
        "SpO2": 95, "SBP": 125, "RR": 20, "Temp": 37.0,
        "dolore_toracico": 1, "dispnea": 1, "alterazione_coscienza": 0,
        "trauma_magg": 0, "sanguinamento_massivo": 0
    }

    # Caso 3: febbre alta senza allarmi → Verde
    case3 = {
        "SpO2": 97, "SBP": 118, "RR": 16, "Temp": 39.2,
        "dolore_toracico": 0, "dispnea": 0, "alterazione_coscienza": 0,
        "trauma_magg": 0, "sanguinamento_massivo": 0
    }

    # Caso 4: trauma maggiore → Rosso
    case4 = {
        "SpO2": 96, "SBP": 130, "RR": 16, "Temp": 36.8,
        "dolore_toracico": 0, "dispnea": 0, "alterazione_coscienza": 0,
        "trauma_magg": 1, "sanguinamento_massivo": 0
    }

    # Esegui
    run_case("Ipossia (Rosso)", case1)
    run_case("Torace+Dispnea (>=Giallo)", case2)
    run_case("Febbre alta (Verde)", case3)
    run_case("Trauma (Rosso)", case4)
