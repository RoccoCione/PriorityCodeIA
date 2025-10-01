# app_streamlit.py
import os, sys
sys.path.append(os.path.dirname(__file__))

import streamlit as st
from datetime import datetime

from src.features import preprocess_input
from src.rules_engine import forward_chain, explain_rules
from src.priority_queue import TriageQueue, SEVERITY_ORDER

# ------------------ CONFIG ------------------
st.set_page_config(page_title="Triage AI (demo)", page_icon="🩺", layout="wide")

TRIAGE_COLORS = {
    "Rosso":  "#e74c3c",
    "Giallo": "#f1c40f",
    "Verde":  "#2ecc71",
    "Bianco": "#bdc3c7",
}

# ------------------ SESSION STATE ------------------
if "queue" not in st.session_state:
    st.session_state.queue = TriageQueue()

q: TriageQueue = st.session_state.queue

# ------------------ HELPERS ------------------
def tri_select(label: str, key: str):
    """
    Ritorna 1 / 0 / None per preprocess_input (yes/no/unknown)
    """
    v = st.selectbox(label, ["No", "Sì", "Incerto"], index=0, key=key)
    return 1 if v == "Sì" else (0 if v == "No" else None)

def counter_badge(label: str, count: int, color: str):
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:10px;border:1px solid #eee;border-radius:10px;padding:10px 14px;">
            <span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:{color};"></span>
            <div>
                <div style="font-weight:600">{label}</div>
                <div style="font-size:22px;font-weight:700;line-height:1">{count}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def text_badge(text: str):
    bg = TRIAGE_COLORS.get(text, "#bdc3c7")
    st.markdown(
        f"""
        <span style="display:inline-block;background:{bg};padding:6px 12px;border-radius:8px;
                     font-weight:700;border:1px solid rgba(0,0,0,0.05)">
            {text}
        </span>
        """,
        unsafe_allow_html=True
    )

def counts_by_class():
    counts = {k: 0 for k in SEVERITY_ORDER}
    for p in q.snapshot():
        counts[p.triage] += 1
    return counts

# ------------------ HEADER ------------------
st.title("🩺 Triage Medico Semplificato — Demo")
st.caption("Progetto didattico: agente a regole + coda a priorità. Non per uso clinico.")

# ------------------ COUNTERS ------------------
counts = counts_by_class()
c1, c2, c3, c4 = st.columns(4)
with c1: counter_badge("Rosso", counts.get("Rosso", 0), TRIAGE_COLORS["Rosso"])
with c2: counter_badge("Giallo", counts.get("Giallo", 0), TRIAGE_COLORS["Giallo"])
with c3: counter_badge("Verde", counts.get("Verde", 0), TRIAGE_COLORS["Verde"])
with c4: counter_badge("Bianco", counts.get("Bianco", 0), TRIAGE_COLORS["Bianco"])

st.divider()

# ------------------ INPUT FORM ------------------
st.subheader("Nuovo paziente")

with st.form("triage_form", clear_on_submit=False):
    cA, cB = st.columns([2, 1])

    with cA:
        name = st.text_input("Nome/etichetta (opzionale)", value="", placeholder="es. Mario R.", key="name")

        st.markdown("**Parametri vitali**")
        v1, v2, v3, v4 = st.columns(4)
        with v1:
            spo2_val = st.number_input("SpO₂ (%)", min_value=70, max_value=100, value=96, step=1, key="spo2_val")
            spo2_nd  = st.checkbox("ND", value=False, key="spo2_nd")
        with v2:
            sbp_val = st.number_input("SBP (mmHg)", min_value=60, max_value=220, value=120, step=1, key="sbp_val")
            sbp_nd  = st.checkbox("ND", value=False, key="sbp_nd")
        with v3:
            rr_val  = st.number_input("RR (atti/min)", min_value=6, max_value=60, value=18, step=1, key="rr_val")
            rr_nd   = st.checkbox("ND", value=False, key="rr_nd")
        with v4:
            temp_val = st.number_input("Temperatura (°C)", min_value=34.0, max_value=42.0, value=37.2, step=0.1, format="%.1f", key="temp_val")
            temp_nd  = st.checkbox("ND", value=False, key="temp_nd")

        # Applica ND (None) se spuntato
        spo2 = None if spo2_nd else int(spo2_val)
        sbp  = None if sbp_nd  else int(sbp_val)
        rr   = None if rr_nd   else int(rr_val)
        temp = None if temp_nd else float(temp_val)

        st.markdown("**Sintomi**")
        s1, s2, s3, s4, s5 = st.columns(5)
        with s1: dolore_toracico = tri_select("Dolore toracico", "dolore_toracico")
        with s2: dispnea = tri_select("Dispnea", "dispnea")
        with s3: alterazione_coscienza = tri_select("Alterazione coscienza", "alterazione_coscienza")
        with s4: trauma_magg = tri_select("Trauma maggiore", "trauma_magg")
        with s5: sanguinamento_massivo = tri_select("Sanguinamento massivo", "sanguinamento_massivo")

    with cB:
        st.info("Compila i campi e premi **Inserisci in coda** per calcolare il triage, vedere le regole attivate e la posizione in coda.")
        submitted = st.form_submit_button("➕ Inserisci in coda")

    if submitted:
        raw = {
            "name": name.strip() or "Anonimo",
            "SpO2": spo2, "SBP": sbp, "RR": rr, "Temp": temp,
            "dolore_toracico": dolore_toracico,
            "dispnea": dispnea,
            "alterazione_coscienza": alterazione_coscienza,
            "trauma_magg": trauma_magg,
            "sanguinamento_massivo": sanguinamento_massivo,
        }

        # Backend: preprocess + regole
        facts = preprocess_input(raw)
        triage, fired, facts_out = forward_chain(facts)

        # Enqueue in coda con payload
        patient = q.enqueue(
            triage=triage,
            payload={"name": raw["name"], "facts": facts_out, "rules": fired}
        )
        pos = q.get_position(patient.patient_id)

        st.success(f"Triage assegnato: {triage}")
        text_badge(triage)
        st.write(f"**Posizione in coda:** {pos} su {len(q)}")
        st.write("**Regole attivate:**")
        st.code(explain_rules(fired), language="text")

        # Aggiorna counters in alto
        counts = counts_by_class()

st.divider()

# ------------------ CODA A PRIORITÀ ------------------
st.subheader("Coda a priorità (globale)")

# Azioni sulla coda
left, mid, right = st.columns([1, 1, 2])
with left:
    if st.button("▶️ Servi prossimo"):
        served = q.serve_next()
        if served:
            st.success(f"Servito: {served.patient_id} | {served.triage} | {served.payload.get('name','')}")
        else:
            st.info("Coda vuota.")
with mid:
    # Select ID da rimuovere
    ids = [p.patient_id for p in q.snapshot()]
    rid = st.selectbox("ID da rimuovere", options=[""] + ids, index=0, key="remove_id")
    if st.button("🗑️ Rimuovi selezionato"):
        if rid:
            removed = q.remove(rid)
            if removed:
                st.success(f"Rimosso: {removed.patient_id} | {removed.triage} | {removed.payload.get('name','')}")
            else:
                st.warning("ID non trovato.")
        else:
            st.warning("Seleziona un ID prima di rimuovere.")
with right:
    if st.button("🧹 Svuota coda"):
        st.session_state.queue = TriageQueue()  # ricrea
        q = st.session_state.queue
        st.info("Coda svuotata.")

# Tabella coda + dettagli
snap = q.snapshot()
if not snap:
    st.write("Nessun paziente in coda.")
else:
    # Mostra tabellina ordinata (già ordinata da snapshot)
    rows = []
    for i, p in enumerate(snap, start=1):
        rows.append({
            "Pos": i,
            "ID": p.patient_id,
            "Nome": p.payload.get("name", ""),
            "Triage": p.triage,
            "Arrivo (UTC)": datetime.utcfromtimestamp(p.arrival_ts).strftime("%H:%M:%S"),
        })
    st.table(rows)

    # Dettagli/Spiegazione per un ID
    sel = st.selectbox("Dettagli regole per ID", options=[""] + [p.patient_id for p in snap], index=0, key="detail_id")
    if sel:
        pp = next((p for p in snap if p.patient_id == sel), None)
        if pp:
            st.markdown(f"**{pp.patient_id}** — {pp.payload.get('name','')}")
            text_badge(pp.triage)
            st.write("**Regole attivate:**")
            st.code(explain_rules(pp.payload.get("rules", [])), language="text")
            with st.expander("Fatti (features)"):
                st.json(pp.payload.get("facts", {}))
