# src/priority_queue.py
"""
Coda a priorità per il triage:
- Ordine di severità: Rosso > Giallo > Verde > Bianco.
- Tie-break: timestamp di arrivo (prima arriva, prima viene servito).
- API principali:
    - TriageQueue.enqueue(patient) -> position
    - TriageQueue.get_position(patient_id) -> int | None
    - TriageQueue.snapshot() -> lista dei pazienti in ordine di servizio
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time
import itertools

SEVERITY_ORDER = ["Bianco", "Verde", "Giallo", "Rosso"]
SEVERITY_RANK = {lvl: i for i, lvl in enumerate(SEVERITY_ORDER)}  # Bianco=0 ... Rosso=3

# generatore di ID univoci leggibili
_id_counter = itertools.count(1)
def new_patient_id(prefix: str = "P") -> str:
    return f"{prefix}{next(_id_counter):04d}"

@dataclass(order=True)
class Patient:
    # L'ordine dataclass non lo usiamo direttamente (ordiniamo noi), ma serve per confronti/sort di sicurezza
    sort_index: int = field(init=False, repr=False, compare=True, default=0)
    patient_id: str
    triage: str                    # "Rosso" | "Giallo" | "Verde" | "Bianco"
    arrival_ts: float              # time.time()
    payload: dict = field(default_factory=dict)  # opzionale: features, spiegazione, ecc.

    def __post_init__(self):
        # sort_index = chiave compatta per un ordinamento globale coerente (rank, ts)
        self.sort_index = (SEVERITY_RANK[self.triage], self.arrival_ts)

class TriageQueue:
    def __init__(self):
        # Liste separate per classe: mantengono l'ordine di arrivo
        self.queues: Dict[str, List[Patient]] = {lvl: [] for lvl in SEVERITY_ORDER}

    # ---------- Metodi interni ----------
    def _count_more_urgent(self, triage: str) -> int:
        """Quanti pazienti sono davanti perché di classe più urgente."""
        my_rank = SEVERITY_RANK[triage]
        return sum(len(self.queues[lvl]) for lvl, r in SEVERITY_RANK.items() if r > my_rank)

    def _position_within_same(self, triage: str, arrival_ts: float) -> int:
        """Quanti pazienti nella stessa classe sono arrivati prima di me."""
        same_list = self.queues[triage]
        return sum(1 for p in same_list if p.arrival_ts <= arrival_ts)

    def _ordered_iter(self):
        """Iteratore dei pazienti in ordine di servizio (globale)."""
        # Dalle classi più urgenti alle meno urgenti; dentro la classe, per timestamp
        for lvl in reversed(SEVERITY_ORDER):
            for p in sorted(self.queues[lvl], key=lambda x: x.arrival_ts):
                yield p

    # ---------- API ----------
    def enqueue(self, triage: str, payload: Optional[dict] = None, patient_id: Optional[str] = None) -> Patient:
        """Inserisce un paziente e ritorna l'oggetto Patient (puoi leggerne la posizione con get_position)."""
        pid = patient_id or new_patient_id()
        ts = time.time()
        patient = Patient(patient_id=pid, triage=triage, arrival_ts=ts, payload=payload or {})
        self.queues[triage].append(patient)
        return patient

    def get_position(self, patient_id: str) -> Optional[int]:
        """
        Ritorna la posizione (1-based) del paziente nella coda globale,
        oppure None se non trovato.
        """
        pos = 1
        for p in self._ordered_iter():
            if p.patient_id == patient_id:
                return pos
            pos += 1
        return None

    def snapshot(self) -> List[Patient]:
        """Lista dei pazienti nell'ordine in cui verranno serviti (globale)."""
        return list(self._ordered_iter())

    def __len__(self) -> int:
        return sum(len(v) for v in self.queues.values())

    def pretty_print(self) -> str:
        """
        Restituisce una stringa multi-linea con la coda globale.
        Formato: [pos] patient_id  |  triage  |  eta_arrivo
        """
        lines = []
        for i, p in enumerate(self._ordered_iter(), start=1):
            lines.append(f"[{i:02d}] {p.patient_id:<6} | {p.triage:<6} | ts={int(p.arrival_ts)}")
        return "\n".join(lines) if lines else "(coda vuota)"

    def remove(self, patient_id: str) -> Optional[Patient]:
        """
        Rimuove un paziente dato il suo ID, indipendentemente dalla posizione o classe.
        Ritorna l'oggetto Patient rimosso, oppure None se non trovato.
        """
        for lvl in SEVERITY_ORDER:
            bucket = self.queues[lvl]
            for idx, p in enumerate(bucket):
                if p.patient_id == patient_id:
                    return bucket.pop(idx)


        # ---------- NEW (opzionale): servi il prossimo ----------

    def serve_next(self) -> Optional[Patient]:
        """
        Estrae e ritorna il prossimo paziente da servire secondo le priorità.
        Ritorna None se la coda è vuota.
        """
        for lvl in reversed(SEVERITY_ORDER):  # Rosso -> ... -> Bianco
            if self.queues[lvl]:
                # nella classe, il primo in ordine di arrivo
                bucket = self.queues[lvl]
                earliest_idx = min(range(len(bucket)), key=lambda i: bucket[i].arrival_ts)
                return bucket.pop(earliest_idx)
        return None
