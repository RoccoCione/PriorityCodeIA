# src/naive_bayes.py
"""
Naive Bayes per il Triage — Versione definitiva (commentata passo-passo)
=======================================================================
Obiettivo:
- Leggere un dataset CATEGORIALE PREESISTENTE (data/examples.csv).
- Allenare Bernoulli Naive Bayes su quello (con pesi anti-sbilanciamento).
- Fornire predizione delle probabilità P(y|x) e decisione "prudente" (costo atteso).
- Esportare un modello salvato in data/nb_model.pkl e ricaricabile.

Nota: questa versione NON genera dataset sintetici. Se il CSV non esiste, solleva un errore chiaro.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import csv
import os
import pickle
from collections import Counter, defaultdict

import numpy as np
from sklearn.naive_bayes import BernoulliNB

# -------------------------------------------------------------------
# 1) SPAZIO DELLE FEATURE (schema) — deve allinearsi al tuo CSV
# -------------------------------------------------------------------
# Ogni chiave ha la lista di categorie ammesse (incluso "unknown").
FEATURE_SPACE: Dict[str, List[str]] = {
    "spo2_cat": ["severa", "moderata", "ok", "unknown"],
    "sbp_cat":  ["severa", "ok", "unknown"],
    "rr_cat":   ["alta", "ok", "unknown"],
    "temp_cat": ["alta", "ok", "unknown"],

    # Sintomi tri-stato
    "dolore_toracico":       ["yes", "no", "unknown"],
    "dispnea":               ["yes", "no", "unknown"],
    "alterazione_coscienza": ["yes", "no", "unknown"],
    "trauma_magg":           ["yes", "no", "unknown"],
    "sanguinamento_massivo": ["yes", "no", "unknown"],
}

# Classi/etichette del triage (ordine crescente di severità)
CLASSES = ["Bianco", "Verde", "Giallo", "Rosso"]
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}

# -------------------------------------------------------------------
# 2) Utility: chiavi one-hot e codifica facts -> vettore binario
# -------------------------------------------------------------------
def _all_onehot_keys() -> List[str]:
    keys = []
    for feat, vals in FEATURE_SPACE.items():
        for v in vals:
            keys.append(f"{feat}__{v}")
    return keys

ONEHOT_KEYS = _all_onehot_keys()
KEY_INDEX = {k: i for i, k in enumerate(ONEHOT_KEYS)}

def encode_onehot(facts: Dict[str, str]) -> np.ndarray:
    """
    Input: facts categoriali (dizionario), es. {"spo2_cat":"ok","dispnea":"yes", ...}
    Output: vettore binario (numpy array) per BernoulliNB.
    Regola: se un valore non è nella lista ammessa -> lo mappiamo a "unknown".
    """
    x = np.zeros(len(ONEHOT_KEYS), dtype=np.int8)
    for feat, vals in FEATURE_SPACE.items():
        v = facts.get(feat, "unknown")
        if v not in vals:
            v = "unknown"
        idx = KEY_INDEX[f"{feat}__{v}"]
        x[idx] = 1
    return x

def max_severity(a: str, b: str) -> str:
    """
    Ritorna la classe più severa tra a e b (usiamo l'ordine in CLASSES).
    """
    return a if CLASSES.index(a) >= CLASSES.index(b) else b

# -------------------------------------------------------------------
# 3) Matrice dei COSTI — decisione "prudente" (costo atteso)
# -------------------------------------------------------------------
# DEFAULT_COST[(y_true, y_pred)] = quanto ci costa quell'errore.
DEFAULT_COST: Dict[Tuple[str, str], float] = {}
def _init_default_cost():
    # di default: costo 1 per qualsiasi errore, 0 se corretto
    for ytrue in CLASSES:
        for yhat in CLASSES:
            DEFAULT_COST[(ytrue, yhat)] = 0.0 if ytrue == yhat else 1.0
    # personalizziamo: sottostimare Rosso costa moltissimo
    DEFAULT_COST[("Rosso", "Giallo")] = 100
    DEFAULT_COST[("Rosso", "Verde")]  = 200
    DEFAULT_COST[("Rosso", "Bianco")] = 500
    # sottostimare Giallo costa intermedio
    DEFAULT_COST[("Giallo", "Verde")]  = 20
    DEFAULT_COST[("Giallo", "Bianco")] = 40
    # falsi positivi "verso l'alto" costano poco
    DEFAULT_COST[("Verde", "Giallo")] = 5
    DEFAULT_COST[("Bianco", "Verde")] = 3
    DEFAULT_COST[("Verde", "Rosso")]  = 8
    DEFAULT_COST[("Bianco", "Rosso")] = 10
_init_default_cost()

def expected_cost(probs: Dict[str, float], yhat: str,
                  cost_matrix: Dict[Tuple[str, str], float] = DEFAULT_COST) -> float:
    """
    Costo atteso di scegliere yhat = somma_y [ P(y|x) * C[y, yhat] ].
    """
    return sum(probs[ytrue] * cost_matrix[(ytrue, yhat)] for ytrue in CLASSES)

def argmin_expected_cost(probs: Dict[str, float],
                         cost_matrix: Dict[Tuple[str, str], float] = DEFAULT_COST) -> str:
    """
    Restituisce la classe con costo atteso minimo (decisione prudente).
    """
    best, best_cost = None, float("inf")
    for yhat in CLASSES:
        c = expected_cost(probs, yhat, cost_matrix)
        if c < best_cost:
            best, best_cost = yhat, c
    return best

# -------------------------------------------------------------------
# 4) Wrapper del modello NB: predizione robusta + decisione cost-sensitive
# -------------------------------------------------------------------
@dataclass
class NBModel:
    """
    Semplice wrapper attorno a BernoulliNB:
    - predizione delle probabilità per tutte le 4 classi (anche se in training ne mancava una)
    - decisione finale "cost-sensitive"
    """
    nb: BernoulliNB
    onehot_keys: List[str] = None
    class_order: List[str] = None

    def predict_proba(self, facts: Dict[str, str]) -> Dict[str, float]:
        """
        facts -> P(y|x) per ogni y in CLASSES.
        Nota: nb.classes_ contiene SOLO le classi viste in training.
              Qui rimappiamo le probabilità su tutte le CLASSES, mettendo 0.0 per le assenti.
        """
        x = encode_onehot(facts).reshape(1, -1)
        p = self.nb.predict_proba(x)[0]  # array lunghezza = numero classi viste
        probs = {c: 0.0 for c in CLASSES}
        for j, cls_idx in enumerate(self.nb.classes_):
            label_name = CLASSES[int(cls_idx)]
            probs[label_name] = float(p[j])
        return probs

    def decide_cost_sensitive(self, facts: Dict[str, str],
                              triage_min: Optional[str] = None,
                              cost_matrix: Dict[Tuple[str, str], float] = DEFAULT_COST
                              ) -> Tuple[str, Dict[str, float]]:
        """
        1) Stima P(y|x).
        2) Sceglie la classe che minimizza il COSTO ATTESO.
        3) Se c'è un 'minimo garantito' (triage_min) dalle regole, non scende sotto quel livello.
        """
        probs = self.predict_proba(facts)
        y = argmin_expected_cost(probs, cost_matrix)
        if triage_min is not None:
            y = max_severity(y, triage_min)
        return y, probs

# -------------------------------------------------------------------
# 5) Fit/Save/Load — allenamento su dataset CATEGORIALE + persistenza
# -------------------------------------------------------------------
def save_model(model: NBModel, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)

def load_model(path: str) -> NBModel:
    with open(path, "rb") as f:
        return pickle.load(f)

def fit_from_dicts(samples: List[Dict[str, str]], labels: List[str]) -> NBModel:
    """
    Allena BernoulliNB da samples categoriali + labels.
    Include una "rete di sicurezza": se nel dataset manca una classe, aggiunge un esempio minimo.
    """
    present = set(labels)
    # booster per classi mancanti (evita modelli "a 3 classi")
    if "Rosso" not in present:
        samples.append({
            "spo2_cat": "severa", "sbp_cat": "ok", "rr_cat": "ok", "temp_cat": "ok",
            "dolore_toracico": "no", "dispnea": "no", "alterazione_coscienza": "no",
            "trauma_magg": "no", "sanguinamento_massivo": "no",
        }); labels.append("Rosso")
    if "Giallo" not in present:
        samples.append({
            "spo2_cat": "ok", "sbp_cat": "ok", "rr_cat": "ok", "temp_cat": "ok",
            "dolore_toracico": "yes", "dispnea": "yes", "alterazione_coscienza": "no",
            "trauma_magg": "no", "sanguinamento_massivo": "no",
        }); labels.append("Giallo")
    if "Verde" not in present:
        samples.append({
            "spo2_cat": "ok", "sbp_cat": "ok", "rr_cat": "ok", "temp_cat": "alta",
            "dolore_toracico": "no", "dispnea": "no", "alterazione_coscienza": "no",
            "trauma_magg": "no", "sanguinamento_massivo": "no",
        }); labels.append("Verde")
    if "Bianco" not in present:
        samples.append({
            "spo2_cat": "ok", "sbp_cat": "ok", "rr_cat": "ok", "temp_cat": "ok",
            "dolore_toracico": "no", "dispnea": "no", "alterazione_coscienza": "no",
            "trauma_magg": "no", "sanguinamento_massivo": "no",
        }); labels.append("Bianco")

    # one-hot + fit
    X = np.vstack([encode_onehot(s) for s in samples])
    y = np.array([CLASS_INDEX[l] for l in labels], dtype=np.int64)

    nb = BernoulliNB()
    nb.fit(X, y)
    return NBModel(nb=nb, onehot_keys=ONEHOT_KEYS, class_order=CLASSES)

# -------------------------------------------------------------------
# 6) Caricamento CSV categoriale e training con validazione
# -------------------------------------------------------------------
def _normalize_label(s: str) -> str:
    s = (s or "").strip()
    mapping = {"bianco":"Bianco","verde":"Verde","giallo":"Giallo","rosso":"Rosso"}
    return mapping.get(s.lower(), s)

def load_dataset_csv(csv_path: str, label_col: str = "label") -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Legge un CSV *categoriale* (già allineato a FEATURE_SPACE).
    - Se trova valori non ammessi per una feature, li mappa a "unknown".
    - Le righe con label non valida vengono scartate.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"[NaiveBayes] CSV non trovato: {csv_path}. "
            f"Assicurati di aver messo il dataset in quella posizione."
        )

    samples, labels = [], []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # facts categoriali conformi a FEATURE_SPACE
            facts = {}
            for feat, vals in FEATURE_SPACE.items():
                v = (row.get(feat, "") or "").strip()
                facts[feat] = v if v in vals else "unknown"

            label = _normalize_label(row.get(label_col, ""))
            if label not in CLASSES:
                # se la label non è valida, saltiamo la riga
                continue

            samples.append(facts)
            labels.append(label)

    if not samples:
        raise ValueError(f"[NaiveBayes] Nessuna riga valida trovata in {csv_path}.")
    return samples, labels

def train_nb_from_csv(csv_path: str,
                      eval_split: float = 0.2,
                      seed: int = 42,
                      model_out: str = "data/nb_model.pkl",
                      use_class_weights: bool = True) -> NBModel:
    """
    - Carica il CSV categoriale.
    - Shuffle + train/val split.
    - (Opzionale) pesi anti-sbilanciamento (inverso-frequenza per classe).
    - Fit BernoulliNB.
    - Valutazione rapida su holdout (accuracy + matrice di confusione).
    - Salva il modello.
    """
    X_facts, y_labels = load_dataset_csv(csv_path)

    # shuffle
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X_facts))
    rng.shuffle(idx)
    X_facts = [X_facts[i] for i in idx]
    y_labels = [y_labels[i] for i in idx]

    # split
    n = len(X_facts)
    n_val = int(n * eval_split)
    X_val, y_val = X_facts[:n_val], y_labels[:n_val]
    X_tr,  y_tr  = X_facts[n_val:], y_labels[n_val:]

    # pesi per classi (inverso-frequenza)
    sample_weight = None
    if use_class_weights:
        cnt = Counter(y_tr)
        inv = {c: 1.0 / max(1, cnt[c]) for c in CLASSES}
        sample_weight = np.array([inv[y] for y in y_tr], dtype=np.float64)

    # fit robusto
    model = fit_from_dicts(X_tr, y_tr)
    if sample_weight is not None:
        X = np.vstack([encode_onehot(s) for s in X_tr])
        y = np.array([CLASS_INDEX[l] for l in y_tr], dtype=np.int64)
        model.nb.fit(X, y, sample_weight=sample_weight)

    # valutazione rapida (argmax della proba, senza costi)
    cm = defaultdict(lambda: defaultdict(int))
    correct = 0
    for facts, ytrue in zip(X_val, y_val):
        p = model.predict_proba(facts)
        yhat = max(p.items(), key=lambda kv: kv[1])[0]
        if yhat == ytrue:
            correct += 1
        cm[ytrue][yhat] += 1
    acc = correct / len(X_val) if X_val else 0.0
    print(f"[VAL] samples={len(X_val)}  accuracy={acc:.3f}")
    print("Confusion matrix (rows=true, cols=pred):")
    header = "        " + " ".join(f"{c:^8}" for c in CLASSES)
    print(header)
    for r in CLASSES:
        row = f"{r:<8} " + " ".join(f"{cm[r][c]:^8}" for c in CLASSES)
        print(row)

    # salva modello
    save_model(model, model_out)
    return model

# -------------------------------------------------------------------
# 7) ensure_trained — carica modello o allena da CSV (niente generazione)
# -------------------------------------------------------------------
def ensure_trained(model_path: str = "data/nb_model.pkl",
                   csv_path: str = "data/examples.csv") -> NBModel:
    """
    Se esiste il modello salvato -> lo carica.
    Altrimenti, allena NAIVE BAYES leggendo il CSV *categoriale* `csv_path`.
    Se il CSV non c'è, solleva un errore (niente generazione automatica).
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    if os.path.exists(model_path):
        return load_model(model_path)

    # allena da CSV preesistente
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"[NaiveBayes] Non trovo il CSV: {csv_path}. "
            f"Metti il tuo dataset categoriale in quella posizione (es. data/examples.csv)."
        )
    print(f"[NB] Training from CSV: {csv_path}")
    return train_nb_from_csv(csv_path, model_out=model_path)
