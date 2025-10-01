# app_streamlit.py
import os, sys
sys.path.append(os.path.dirname(__file__))
import pandas as pd

import streamlit as st
from datetime import datetime

from src.features import preprocess_input
from src.rules_engine import forward_chain, explain_rules
from src.priority_queue import TriageQueue, SEVERITY_ORDER

# -------- CONFIG --------
st.set_page_config(page_title="Triage AI (demo)", page_icon="🩺", layout="wide")
TRIAGE_EMOJI  = {"Rosso":"🔴","Giallo":"🟡","Verde":"🟢","Bianco":"⚪"}
TRIAGE_COLORS = {"Rosso":"#e74c3c","Giallo":"#f1c40f","Verde":"#2ecc71","Bianco":"#bdc3c7"}

# -------- CUSTOM STYLE (bordo/ombre per metric e tabella + card generica) --------
CUSTOM_CSS = """
<style>
/* sfondo leggero della pagina */
section.main > div { background: #f8fafc; }

/* card generica */
.card {
  border: none;
  border-radius: 14px;
  padding: 16px;
  box-shadow: 0 1px 2px rgba(0,0,0,.04);
}

/* metric con bordo e ombra */
div[data-testid="stMetric"] {
  border: 1px solid #ffffff;
  border-radius: 12px;
  padding: 12px 14px;
  box-shadow: 0 1px 2px rgba(0,0,0,.03);
  color: #ffffff;
}
div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
  color: #ffffff !important;
  font-weight: 1000;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
  font-weight: 1000;
}

/* contenitore della tabella con bordo e raggio */
div[data-testid="stDataFrame"] {
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  overflow: hidden;
  background: #ffffff;
  box-shadow: 0 1px 2px rgba(0,0,0,.04);
}

/* badge semplice */
.badge {
  display:inline-block;
  background:#eef2f7;
  padding:6px 10px;
  border-radius:999px;
  font-weight:700;
  border:1px solid #e5e7eb;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -------- SESSION --------
if "queue" not in st.session_state:
    st.session_state.queue = TriageQueue()
if "last_result" not in st.session_state:
    st.session_state.last_result = None

q: TriageQueue = st.session_state.queue

# -------- HELPERS --------
def tri_select(label: str, key: str):
    v = st.selectbox(label, ["No", "Sì", "Incerto"], index=0, key=key)
    return 1 if v == "Sì" else (0 if v == "No" else None)

def counts_by_class():
    counts = {k: 0 for k in SEVERITY_ORDER}
    for p in q.snapshot():
        counts[p.triage] += 1
    return counts

def tri_badge(text: str):
    bg = TRIAGE_COLORS.get(text, "#e5e7eb")
    fg = "#000000" if text in ("Giallo", "Bianco") else "#ffffff"
    st.markdown(
        f"<span class='badge' style='background:{bg};color:{fg};border-color:rgba(0,0,0,0.08)'>{text}</span>",
        unsafe_allow_html=True
    )

# -------- HEADER + CONTATORI --------
st.title("🩺 Triage Medico Semplificato — Demo")
st.caption("Agente a regole + coda a priorità (uso didattico, non clinico).")

counts = counts_by_class()
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("Rosso 🔴", counts["Rosso"])
with c2: st.metric("Giallo 🟡", counts["Giallo"])
with c3: st.metric("Verde 🟢", counts["Verde"])
with c4: st.metric("Bianco ⚪", counts["Bianco"])

st.divider()

# -------- FORM (sx) + RISULTATO (dx) --------
col_form, col_res = st.columns([1.25, 1])

with col_form:

    st.subheader("Nuovo paziente")

    with st.form("triage_form", clear_on_submit=False):
        name = st.text_input("Nome/etichetta", value="", placeholder="es. Mario R.", key="name")

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

        st.markdown("**Sintomi**")
        s1, s2, s3, s4, s5 = st.columns(5)
        with s1: dolore_toracico = tri_select("Dolore toracico", "dolore_toracico")
        with s2: dispnea = tri_select("Dispnea", "dispnea")
        with s3: alterazione_coscienza = tri_select("Alterazione coscienza", "alterazione_coscienza")
        with s4: trauma_magg = tri_select("Trauma maggiore", "trauma_magg")
        with s5: sanguinamento_massivo = tri_select("Sanguinamento massivo", "sanguinamento_massivo")

        col_submit, col_reset = st.columns([1, 1])
        submitted = col_submit.form_submit_button("➕ Inserisci in coda", use_container_width=True)
        reset = col_reset.form_submit_button("↺ Pulisci campi", use_container_width=True)

    if submitted:
        raw = {
            "name": name.strip() or "Anonimo",
            "SpO2": None if spo2_nd else int(spo2_val),
            "SBP" : None if sbp_nd  else int(sbp_val),
            "RR"  : None if rr_nd   else int(rr_val),
            "Temp": None if temp_nd else float(temp_val),
            "dolore_toracico": dolore_toracico,
            "dispnea": dispnea,
            "alterazione_coscienza": alterazione_coscienza,
            "trauma_magg": trauma_magg,
            "sanguinamento_massivo": sanguinamento_massivo,
        }

        facts = preprocess_input(raw)
        triage, fired, facts_out = forward_chain(facts)
        patient = q.enqueue(
            triage=triage,
            payload={"name": raw["name"], "facts": facts_out, "rules": fired}
        )
        pos = q.get_position(patient.patient_id)

        st.session_state.last_result = {
            "triage": triage,
            "pos": pos,
            "queue_len": len(q),
            "rules": fired,
            "facts": facts_out,
            "name": raw["name"],
            "id": patient.patient_id,
        }
        st.rerun()  # aggiorna contatori + tabella subito

    if reset:
        #azzera anche il riquadro “Risultato”
        st.session_state.last_result = None

        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

with col_res:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Risultato")
    lr = st.session_state.last_result
    if not lr:
        st.caption("Nessun risultato ancora. Inserisci un paziente nel pannello a sinistra.")
    else:
        st.write(f"**Paziente:** {lr['name']}  •  **ID:** `{lr['id']}`")
        tri_badge(lr["triage"])
        st.write(f"**Posizione in coda:** {lr['pos']} su {lr['queue_len']}")
        st.write("**Regole attivate:**")
        st.code(explain_rules(lr["rules"]), language="text")
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# -------- CODA: tabella (con bordo) + barra azioni --------
st.subheader("Coda a priorità (globale)")

snap = q.snapshot()
if not snap:
    st.write("Nessun paziente in coda.")
else:
    # --- CARD: paziente attualmente servito (top della coda) ---
    top = snap[0]
    st.markdown('<div class="card" style="margin-bottom:10px">', unsafe_allow_html=True)
    st.markdown("**▶️ In servizio ora**")
    st.write(f"**ID** `{top.patient_id}` — {top.payload.get('name','')}")
    tri_badge(top.triage)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Tabella con colonna 'In servizio' e riga evidenziata ---
    rows = []
    for i, p in enumerate(snap, start=1):
        rows.append({
            "Pos": i,
            "In servizio": "▶️" if i == 1 else "",
            "ID": p.patient_id,
            "Nome": p.payload.get("name", ""),
            "Triage": f"{TRIAGE_EMOJI[p.triage]} {p.triage}",
            "Arrivo (UTC)": datetime.utcfromtimestamp(p.arrival_ts).strftime("%H:%M:%S"),
        })
    df = pd.DataFrame(rows)


    def _opaque_top(row):
        # rende la PRIMA riga semitrasparente (opaca)
        return ['opacity: 0.55' if row.name == 0 else '' for _ in row]


    st.dataframe(
        df.style.apply(_opaque_top, axis=1),
        use_container_width=True,
        hide_index=True
    )

    # --- BARRA AZIONI (stessa linea) ---
    colA, colB1, colB2, colC = st.columns([1, 2, 1, 1])
    with colA:
        if st.button("▶️ Servi prossimo", use_container_width=True):
            served = q.serve_next()
            if served:
                st.success(f"Servito: {served.patient_id} | {served.triage} | {served.payload.get('name','')}")
            else:
                st.info("Coda vuota.")
            st.rerun()

    with colB1:
        ids = [p.patient_id for p in snap]
        rid = st.selectbox("ID da rimuovere", options=[""] + ids, index=0, key="remove_id_inline")
    with colB2:
        if st.button("🗑️ Rimuovi selezionato", use_container_width=True):
            if rid:
                removed = q.remove(rid)
                if removed:
                    st.success(f"Rimosso: {removed.patient_id} | {removed.triage} | {removed.payload.get('name','')}")
                else:
                    st.warning("ID non trovato.")
            else:
                st.warning("Seleziona un ID.")
            st.rerun()

    with colC:
        if st.button("🧹 Svuota coda", use_container_width=True):
            st.session_state.queue = TriageQueue()
            st.session_state.last_result = None
            st.info("Coda svuotata.")
            st.rerun()

    # Dettagli (facoltativo)
    with st.expander("Dettagli paziente (regole + facts)"):
        sel = st.selectbox("Seleziona ID", options=[""] + [p.patient_id for p in q.snapshot()], index=0, key="detail_id")
        if sel:
            pp = next((p for p in q.snapshot() if p.patient_id == sel), None)
            if pp:
                st.write(f"**{pp.patient_id}** — {pp.payload.get('name','')}")
                st.write("**Regole attivate:**")
                st.code(explain_rules(pp.payload.get("rules", [])), language="text")
                st.write("**Fatti (features):**")
                st.json(pp.payload.get("facts", {}))

