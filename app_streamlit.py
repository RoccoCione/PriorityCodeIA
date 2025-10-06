# app_streamlit.py
import os, sys
sys.path.append(os.path.dirname(__file__))
import pandas as pd
import streamlit as st
from datetime import datetime

from src.features import preprocess_input
from src.rules_engine import forward_chain, explain_rules, default_kb
from src.priority_queue import TriageQueue, SEVERITY_ORDER
from src.naive_bayes import ensure_trained, NBModel

# -------- CONFIG --------
st.set_page_config(page_title="Triage AI (demo)", page_icon="🩺", layout="wide")
TRIAGE_EMOJI  = {"Rosso":"🔴","Giallo":"🟡","Verde":"🟢","Bianco":"⚪"}
TRIAGE_COLORS = {"Rosso":"#e74c3c","Giallo":"#f1c40f","Verde":"#2ecc71","Bianco":"#bdc3c7"}

# ---- NB MODEL (una sola volta, in sessione) ----
if "nb_model" not in st.session_state:
    st.session_state.nb_model = ensure_trained(
        model_path="data/nb_model.pkl",
        csv_path="data/examples.csv"   # dataset categoriale statico
    )
nb_model: NBModel = st.session_state.nb_model

# -------- CUSTOM STYLE --------
CUSTOM_CSS = """
<style>
section.main > div { background: #f8fafc; }
section.main > div.block-container { padding-top: 1.0rem; }
.card { border: none; border-radius: 14px; padding: 16px; box-shadow: 0 1px 2px rgba(0,0,0,.04); background: #ffffff; }
div[data-testid="stMetric"] { border: 1px solid #ffffff; border-radius: 12px; padding: 12px 14px; box-shadow: 0 1px 2px rgba(0,0,0,.03); color: #ffffff; align-items: flex-end; }
div[data-testid="stMetric"] > div { padding-bottom: 0 !important; }
div[data-testid="stMetricValue"] { line-height: 1; }
div[data-testid="stMetric"] [data-testid="stMetricLabel"] { color: #ffffff !important; font-weight: 1000; margin-bottom: 0.15rem; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-weight: 1000; }
div[data-testid="stDataFrame"] { border: 1px solid #e5e7eb; border-radius: 12px; overflow: hidden; background: #ffffff; box-shadow: 0 1px 2px rgba(0,0,0,.04); }
.badge { display:inline-block; background:#eef2f7; padding:6px 10px; border-radius:999px; font-weight:700; border:1px solid #e5e7eb; }
.serving { position: relative; border-radius: 14px; padding: 14px 16px 12px 16px; background: transparent; display: grid; grid-template-columns: 1fr auto; row-gap: 8px; margin-bottom: 12px; color: #ffffff; font-size: 1.1rem; }
.serving::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 6px; border-top-left-radius: 14px; border-bottom-left-radius: 14px; background: var(--serving-accent, #94a3b8); }
.serving .title { font-weight: 700; font-size: 1.2rem; display: flex; align-items: center; gap: 6px; }
.serving .meta { font-size: 1rem; color: #ffffff; }
html, body, [class^="css"], [class*="css"] { font-size: 1.1rem; }
h1, .stMarkdown h1, div[data-testid="stHeader"] { font-size: 2.0rem !important; }
h2, .stMarkdown h2 { font-size: 1.6rem !important; }
h3, .stMarkdown h3 { font-size: 1.3rem !important; }
code, pre, .stCode { font-size: 0.95rem !important; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { padding: 6px 10px; border-radius: 10px; border: 1px solid #e5e7eb; box-shadow: 0 1px 2px rgba(0,0,0,.03); }
/* --- Probabilità NB: barre orizzontali --- */
.prob-box { margin-top: 6px; }
.prob-row {
  display: grid; grid-template-columns: 130px 1fr 64px; align-items: center;
  gap: 8px; margin: 6px 0;
}
.prob-label { font-weight: 700; white-space: nowrap; }
.prob-bar {
  position: relative; height: 12px; border-radius: 999px; background: #f1f5f9; overflow: hidden;
  box-shadow: inset 0 0 0 1px rgba(0,0,0,.05);
}
.prob-fill {
  position: absolute; left: 0; top: 0; bottom: 0; width: var(--w, 0%);
  background: var(--c, #94a3b8); transition: width .25s ease;
}
.prob-val { text-align: right; font-variant-numeric: tabular-nums; font-weight: 700; }

/* --- DEBUG panel styling --- */
.debug-box {
   border-radius: 12px;
  padding: 12px 14px; box-shadow: 0 1px 2px rgba(0,0,0,.04);
}
.debug-grid { display: grid; grid-template-columns: 170px 1fr; gap: 8px 12px; align-items: center; }
.debug-k { font-weight: 700; color: #475569; }
.chips { display:flex; flex-wrap:wrap; gap:6px; }
.chip {
  background:#f1f5f9; border:1px solid #e5e7eb; border-radius:999px; padding:4px 8px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
  font-size: 0.9rem;
}
.soft-hr { height:1px; background:#e5e7eb; margin:10px 0; }

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

def compute_triage_min_from_fired(fired_ids):
    """
    Legge la KB per capire se alcune regole attivate impongono un 'triage_min'
    (es. 'almeno Giallo'). Ritorna None se non ci sono vincoli.
    """
    kb = {r.id: r for r in default_kb()}
    order = ["Bianco","Verde","Giallo","Rosso"]
    def max_sev(a,b): return a if order.index(a) >= order.index(b) else b
    tri_min = None
    for rid in fired_ids:
        r = kb.get(rid)
        if r and "triage_min" in r.then:
            tri_min = r.then["triage_min"] if tri_min is None else max_sev(tri_min, r.then["triage_min"])
    return tri_min

def render_prob_bars(probs: dict):
    order = ["Rosso", "Giallo", "Verde", "Bianco"]  # ordine clinico
    rows = []
    for cls in order:
        p = float(probs.get(cls, 0.0))  # si assume già numerico e tra 0 e 1
        pct = max(0.0, min(100.0, round(p * 100, 1)))
        color = TRIAGE_COLORS.get(cls, "#94a3b8")
        label = f"{TRIAGE_EMOJI[cls]} {cls}"
        width_pct = pct if pct > 0 else 0
        if 0 < width_pct < 2:
            width_pct = 2.0  # traccia minima per valori molto piccoli
        row_html = (
            '<div class="prob-row">'
            f'<div class="prob-label">{label}</div>'
            '<div class="prob-bar">'
            f'<div class="prob-fill" style="width:{width_pct}%; background:{color};"></div>'
            '</div>'
            f'<div class="prob-val">{pct:.1f}%</div>'
            '</div>'
        )
        rows.append(row_html)
    st.markdown('<div class="prob-box">' + "".join(rows) + "</div>", unsafe_allow_html=True)


# -------- HEADER + CONTATORI --------
st.title("🩺 Triage Medico Semplificato — Demo")
st.caption("Agente a regole + Naive Bayes (prudente) + coda a priorità. Uso didattico, non clinico.")

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
        reset = col_reset.form_submit_button("↺ Pulisci risultato", use_container_width=True)

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

        # ---- Preprocess -> Regole -> (Rosso? stop) / altrimenti NB cost-sensitive ----
        facts = preprocess_input(raw)
        triage_rules, fired, facts_out = forward_chain(facts)

        if triage_rules == "Rosso":
            final_triage = "Rosso"
            probs = {"Bianco":0.0,"Verde":0.0,"Giallo":0.0,"Rosso":1.0}
        else:
            triage_min = compute_triage_min_from_fired(fired)  # es. "Giallo" come minimo garantito
            final_triage, probs = nb_model.decide_cost_sensitive(facts_out, triage_min=triage_min)

        # ---- Inserisci in coda + salva risultato in sessione ----
        patient = q.enqueue(
            triage=final_triage,
            payload={"name": raw["name"], "facts": facts_out, "rules": fired, "probs": probs}
        )
        pos = q.get_position(patient.patient_id)

        st.session_state.last_result = {
            "triage": final_triage,
            "pos": pos,
            "queue_len": len(q),
            "rules": fired,
            "facts": facts_out,
            "name": raw["name"],
            "id": patient.patient_id,
            "probs": probs
        }
        st.rerun()

    if reset:
        st.session_state.last_result = None
        st.rerun()

with col_res:
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
        if "probs" in lr and lr["probs"]:
            st.write("**Probabilità (Naive Bayes):**")
            render_prob_bars(lr["probs"])
st.divider()

# -------- CODA: tabella + barra azioni --------
st.subheader("Attualmente in servizio")

snap = q.snapshot()
if not snap:
    st.write("Nessun paziente al pronto soccorso.")
else:
    # --- CARD: paziente attualmente servito (top della coda) ---
    top = snap[0]
    tri_col = TRIAGE_COLORS.get(top.triage, "#94a3b8")
    waited_sec = max(0, int(datetime.utcnow().timestamp() - top.arrival_ts))
    mm, ss = divmod(waited_sec, 60)
    waited_str = f"{mm:02d}:{ss:02d}"

    st.markdown(
        f"""
        <div class="serving" style="--serving-accent:{tri_col}">
          <div class="title">▶️ In servizio ora</div>
          <div></div>
          <div class="meta"><strong>ID:</strong> <code>{top.patient_id}</code> — {top.payload.get('name', '')}</div>
          <div>
            <span class="badge" style="background:{tri_col};color:#000000;border-color:rgba(0,0,0,0)">
                {top.triage}
            </span>
          </div>
          <div class="meta">
            Arrivo (UTC): {datetime.utcfromtimestamp(top.arrival_ts).strftime("%H:%M:%S")} • Attesa: {waited_str}
          </div>
          <div></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.subheader("Coda a priorità (globale)")
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
        return ['opacity: 0.55' if row.name == 0 else '' for _ in row]

    st.dataframe(
        df.style.apply(_opaque_top, axis=1),
        use_container_width=True,
        hide_index=True
    )

    st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)

    # --- BARRA AZIONI ---
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

    # Dettagli
    with st.expander("Dettagli paziente (regole + facts)"):
        ids_opts = [""] + [p.patient_id for p in q.snapshot()]
        sel = st.selectbox("Seleziona ID", options=ids_opts, index=0, key="detail_id")
        if sel:
            pp = next((p for p in q.snapshot() if p.patient_id == sel), None)
            if pp:
                st.divider()
                h1, h2 = st.columns([1, 1])
                with h1:
                    st.markdown(f"**ID:** <code>{pp.patient_id}</code>", unsafe_allow_html=True)
                    st.markdown(f"**Nome:** {pp.payload.get('name', '')}")

                with h2:
                    tri_badge(pp.triage)
                    st.markdown(f"**Arrivo (UTC):** {datetime.utcfromtimestamp(pp.arrival_ts).strftime('%H:%M:%S')}")

                st.markdown("---")

                tab_rules, tab_facts, tab_probs = st.tabs(["Regole attivate", "Fatti (features)", "Probabilità NB"])
                with tab_rules:
                    rules_text = explain_rules(pp.payload.get("rules", []))
                    st.code(rules_text or "Nessuna regola attivata.", language="text")

                with tab_facts:
                    facts_dict = pp.payload.get("facts", {})
                    if facts_dict:
                        facts_rows = [{"Feature": k, "Valore": v} for k, v in sorted(facts_dict.items())]
                        facts_df = pd.DataFrame(facts_rows)
                        try:
                            st.table(facts_df.style.hide_index())
                        except Exception:
                            st.dataframe(facts_df, use_container_width=True, hide_index=True)
                    else:
                        st.caption("Nessun fact disponibile.")

                with tab_probs:
                    probs = pp.payload.get("probs", {})
                    if probs:
                        st.markdown("**Probabilità (Naive Bayes)**")
                        render_prob_bars(probs)  # usa lo stesso stile del pannello Risultato
                        with st.expander("Mostra valori"):
                            st.json({k: round(v, 3) for k, v in probs.items()})
                    else:
                        st.caption("Nessuna probabilità salvata per questo paziente.")

# ---------- DEBUG DIAGNOSTICO ----------
with st.expander("🔎 DEBUG pipeline (ultimo inserito)"):
    lr = st.session_state.last_result
    if not lr:
        st.caption("Inserisci prima un paziente.")
    else:
        # ricalcolo regole e vincolo
        triage_rules_dbg, fired_dbg, _facts_out_dbg = forward_chain(lr["facts"])
        kb_map = {r.id: r for r in default_kb()}
        order = ["Bianco","Verde","Giallo","Rosso"]
        def max_sev(a,b): return a if order.index(a) >= order.index(b) else b
        tri_min_dbg = None
        for rid in fired_dbg:
            r = kb_map.get(rid)
            if r and "triage_min" in r.then:
                tri_min_dbg = r.then["triage_min"] if tri_min_dbg is None else max_sev(tri_min_dbg, r.then["triage_min"])

        # Header
        st.markdown(
            '<div class="debug-box">'
            '<div class="debug-grid">'
            f'<div class="debug-k">Paziente</div><div><code>{lr["id"]}</code> — {lr["name"]}</div>'
            f'<div class="debug-k">Codice finale</div><div>', unsafe_allow_html=True
        )
        tri_badge(lr["triage"])
        st.markdown('</div>', unsafe_allow_html=True)  # chiude la cella della grid
        st.markdown(
            f'<div class="debug-grid">'
            f'<div class="debug-k">Da regole</div><div>{triage_rules_dbg}</div>'
            f'<div class="debug-k">Vincolo (triage_min)</div><div>{tri_min_dbg if tri_min_dbg else "—"}</div>'
            f'<div class="debug-k">Posizione in coda</div><div>{lr["pos"]} / {lr["queue_len"]}</div>'
            '</div>'
            '<div class="soft-hr"></div>',
            unsafe_allow_html=True
        )

        # Regole attivate (chips)
        chips_html = ''.join(f'<span class="chip">{rid}</span>' for rid in fired_dbg)
        st.markdown(f"**Regole attivate (IDs):**<div class='chips'>{chips_html or '—'}</div>", unsafe_allow_html=True)

        # Probabilità
        probs_dbg = lr.get("probs") or {}
        if isinstance(probs_dbg, dict) and probs_dbg:
            st.markdown("**Probabilità (Naive Bayes)**")
            render_prob_bars(probs_dbg)
        else:
            st.caption("Nessuna probabilità NB salvata per l'ultimo inserito.")

        st.markdown('<div class="soft-hr"></div>', unsafe_allow_html=True)

        # Facts
        facts_dict = lr.get("facts", {})
        st.markdown("**Facts (dopo preprocess)**")
        if facts_dict:
            facts_df = pd.DataFrame(
                [{"Feature": k, "Valore": v} for k, v in sorted(facts_dict.items())]
            )
            try:
                st.table(facts_df.style.hide_index())
            except Exception:
                st.dataframe(facts_df, use_container_width=True, hide_index=True)
        else:
            st.caption("Nessun fact disponibile.")

        # raw objects
        with st.expander("Raw objects (debug avanzato)"):
            st.write("last_result:")
            st.json(lr)
