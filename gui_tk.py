# gui_tk.py
# GUI Tkinter migliorata:
# - badge conteggio per classe (Rosso/Giallo/Verde/Bianco)
# - tema 'clam', layout più pulito, badge colorati
# - righe della coda colorate per classe tramite tag Treeview
# - azioni: inserisci, servi prossimo, rimuovi selezionato, svuota coda

import os, sys
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime

# assicurati che src/ sia importabile
sys.path.append(os.path.dirname(__file__))

from src.features import preprocess_input
from src.rules_engine import forward_chain, explain_rules
from src.priority_queue import TriageQueue, SEVERITY_ORDER

# Colori principali per badge del codice
TRIAGE_COLORS = {
    "Rosso":  "#e74c3c",
    "Giallo": "#f1c40f",
    "Verde":  "#2ecc71",
    "Bianco": "#bdc3c7",
}
# Colori tenui per colorare le righe della tabella
ROW_SHADE = {
    "Rosso":  "#fde2e0",
    "Giallo": "#fff5d6",
    "Verde":  "#e9f7ef",
    "Bianco": "#f2f4f5",
}

class TriageApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Triage AI — Demo (Tkinter)")
        self.geometry("980x720")
        self.minsize(940, 680)

        # Tema un po' più moderno
        try:
            ttk.Style().theme_use("clam")
        except Exception:
            pass

        self.queue = TriageQueue()
        self._build_ui()

    # ---------------- UI ----------------
    def _build_ui(self):
        root = ttk.Frame(self, padding=16)
        root.pack(fill="both", expand=True)

        # Header
        head = ttk.Frame(root)
        head.pack(fill="x", pady=(0, 8))
        title = ttk.Label(head, text="Triage Medico Semplificato", font=("Segoe UI", 16, "bold"))
        subtitle = ttk.Label(head, text="Valutazione con regole + coda a priorità (demo didattica)", foreground="#555")
        title.pack(anchor="w")
        subtitle.pack(anchor="w")

        # --- COUNTERS (badge per classe) ---
        counters = ttk.Frame(root)
        counters.pack(fill="x", pady=(4, 12))
        self.badge_frames = {}
        for name in ["Rosso", "Giallo", "Verde", "Bianco"]:
            f = self._make_counter_badge(counters, name, TRIAGE_COLORS[name])
            f.pack(side="left", padx=6)
            self.badge_frames[name] = f

        # --- INPUT ---
        lf_in = ttk.LabelFrame(root, text="Dati paziente")
        lf_in.pack(fill="x")

        # riga 1: nome + vitali
        row1 = ttk.Frame(lf_in)
        row1.pack(fill="x", pady=(8, 0))

        self.var_name = tk.StringVar()
        self._labeled_entry(row1, "Nome/etichetta", self.var_name, 0, 0, width=24)

        self.var_spo2 = tk.StringVar()
        self._labeled_entry(row1, "SpO₂ (%)", self.var_spo2, 0, 1)

        self.var_sbp  = tk.StringVar()
        self._labeled_entry(row1, "SBP (mmHg)", self.var_sbp, 0, 2)

        self.var_rr   = tk.StringVar()
        self._labeled_entry(row1, "RR (atti/min)", self.var_rr, 0, 3)

        self.var_temp = tk.StringVar()
        self._labeled_entry(row1, "Temperatura (°C)", self.var_temp, 0, 4)

        # riga 2: sintomi tri-stato
        row2 = ttk.Frame(lf_in)
        row2.pack(fill="x", pady=(8, 8))

        self.symptoms = {}
        for i, label in enumerate([
            "Dolore toracico", "Dispnea",
            "Alterazione coscienza", "Trauma maggiore", "Sanguinamento massivo"
        ]):
            var = tk.StringVar(value="No")
            self.symptoms[label] = var
            self._labeled_tri(row2, label, var, 0, i)

        # pulsanti azione
        row_btn = ttk.Frame(lf_in)
        row_btn.pack(fill="x", pady=(8, 10))
        ttk.Button(row_btn, text="Inserisci in coda", command=self.on_submit)\
            .pack(side="left")
        ttk.Button(row_btn, text="Servi prossimo", command=self.on_serve_next)\
            .pack(side="left", padx=(8, 0))
        ttk.Button(row_btn, text="Rimuovi selezionato", command=self.on_remove_selected)\
            .pack(side="left", padx=(8, 0))
        ttk.Button(row_btn, text="Svuota coda", command=self.on_reset_queue)\
            .pack(side="left", padx=(8, 0))
        ttk.Button(row_btn, text="Pulisci campi", command=self.on_clear_inputs)\
            .pack(side="left", padx=(8, 0))

        # --- RISULTATO CORRENTE ---
        lf_res = ttk.LabelFrame(root, text="Risultato corrente")
        lf_res.pack(fill="x")

        top_res = ttk.Frame(lf_res)
        top_res.pack(fill="x", pady=6)

        ttk.Label(top_res, text="Triage:").pack(side="left")
        self.triage_badge = tk.Label(top_res, text="—", width=10, relief="groove", bg="#bdc3c7")
        self.triage_badge.pack(side="left", padx=(8, 18), ipadx=8, ipady=3)

        ttk.Label(top_res, text="Posizione in coda:").pack(side="left")
        self.pos_lbl = ttk.Label(top_res, text="—")
        self.pos_lbl.pack(side="left", padx=(8, 0))

        # spiegazione regole
        ttk.Label(lf_res, text="Regole attivate:").pack(anchor="w")
        self.rules_text = tk.Text(lf_res, height=8, wrap="word")
        self.rules_text.pack(fill="x", padx=4, pady=(2, 8))
        self.rules_text.configure(state="disabled")

        # --- CODA GLOBALE ---
        lf_queue = ttk.LabelFrame(root, text="Coda a priorità (globale)")
        lf_queue.pack(fill="both", expand=True, pady=(10, 0))

        cols = ("pos", "id", "name", "triage", "arrivo")
        self.tree = ttk.Treeview(lf_queue, columns=cols, show="headings", height=12)
        for c, w, anchor in [
            ("pos", 60, "center"),
            ("id", 90, "center"),
            ("name", 260, "w"),
            ("triage", 110, "center"),
            ("arrivo", 150, "center"),
        ]:
            self.tree.heading(c, text=c.upper())
            self.tree.column(c, width=w, anchor=anchor)
        self.tree.pack(fill="both", expand=True, padx=6, pady=6)

        # Colori riga per codice
        for code, shade in ROW_SHADE.items():
            self.tree.tag_configure(code, background=shade)

        # selezione riga -> mostra regole del paziente, evidenzia badge
        self.tree.bind("<<TreeviewSelect>>", self.on_select_row)

        # aggiornamento iniziale contatori
        self._update_counters()

    # ------- widget helper -------
    def _make_counter_badge(self, parent, label, color):
        # card con “pallino” colorato, etichetta e numero grande
        frame = ttk.Frame(parent, relief="groove", padding=8)
        top = ttk.Frame(frame)
        top.pack(fill="x")
        dot = tk.Canvas(top, width=14, height=14, highlightthickness=0)
        dot.pack(side="left")
        dot.create_oval(2, 2, 12, 12, fill=color, outline=color)
        ttk.Label(top, text=label, font=("Segoe UI", 10, "bold")).pack(side="left", padx=(6, 0))
        count_lbl = ttk.Label(frame, text="0", font=("Segoe UI", 16, "bold"))
        count_lbl.pack(anchor="w", pady=(4, 0))
        # salviamo riferimento per aggiornamento
        frame.count_lbl = count_lbl
        frame.dot = dot
        return frame

    def _labeled_entry(self, parent, text, var, row, col, width=12):
        cell = ttk.Frame(parent)
        cell.grid(row=row, column=col, padx=6, pady=4, sticky="w")
        ttk.Label(cell, text=text).pack(anchor="w")
        ttk.Entry(cell, textvariable=var, width=width).pack(anchor="w")

    def _labeled_tri(self, parent, text, var, row, col):
        cell = ttk.Frame(parent)
        cell.grid(row=row, column=col, padx=6, pady=4, sticky="w")
        ttk.Label(cell, text=text).pack(anchor="w")
        cb = ttk.Combobox(cell, textvariable=var, width=14, state="readonly",
                          values=["No", "Sì", "Incerto"])
        cb.pack(anchor="w")
        cb.current(0)

    # ------- azioni -------
    def on_submit(self):
        name = (self.var_name.get() or "").strip() or "Anonimo"

        def parse_int(s):
            s = (s or "").strip()
            if not s: return None
            try: return int(s)
            except: return None

        def parse_float(s):
            s = (s or "").strip().replace(",", ".")
            if not s: return None
            try: return float(s)
            except: return None

        raw = {
            "name": name,
            "SpO2": parse_int(self.var_spo2.get()),
            "SBP":  parse_int(self.var_sbp.get()),
            "RR":   parse_int(self.var_rr.get()),
            "Temp": parse_float(self.var_temp.get()),
            "dolore_toracico": self._tri_to_01n(self.symptoms["Dolore toracico"].get()),
            "dispnea": self._tri_to_01n(self.symptoms["Dispnea"].get()),
            "alterazione_coscienza": self._tri_to_01n(self.symptoms["Alterazione coscienza"].get()),
            "trauma_magg": self._tri_to_01n(self.symptoms["Trauma maggiore"].get()),
            "sanguinamento_massivo": self._tri_to_01n(self.symptoms["Sanguinamento massivo"].get()),
        }

        facts = preprocess_input(raw)
        triage, fired, facts_out = forward_chain(facts)

        patient = self.queue.enqueue(
            triage=triage,
            payload={"name": name, "facts": facts_out, "rules": fired}
        )
        pos = self.queue.get_position(patient.patient_id)

        self._set_badge(triage)
        self.pos_lbl.config(text=f"{pos} su {len(self.queue)}")
        self._set_rules_text(explain_rules(fired))

        self._refresh_queue_table()

    def on_serve_next(self):
        served = self.queue.serve_next()
        if served:
            messagebox.showinfo("Servito", f"{served.patient_id} | {served.triage} | {served.payload.get('name','')}")
        else:
            messagebox.showinfo("Servito", "Coda vuota.")
        self._refresh_queue_table()

    def on_remove_selected(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showwarning("Rimuovi", "Seleziona una riga nella coda.")
            return
        pid = self.tree.set(sel[0], "id")
        removed = self.queue.remove(pid)
        if removed:
            messagebox.showinfo("Rimosso", f"{removed.patient_id} | {removed.triage} | {removed.payload.get('name','')}")
        else:
            messagebox.showwarning("Rimuovi", "ID non trovato.")
        self._refresh_queue_table()

    def on_reset_queue(self):
        if messagebox.askyesno("Conferma", "Svuotare la coda?"):
            self.queue = TriageQueue()
            self._set_badge("—")
            self.pos_lbl.config(text="—")
            self._set_rules_text("")
            self._refresh_queue_table()

    def on_clear_inputs(self):
        self.var_name.set("")
        self.var_spo2.set("")
        self.var_sbp.set("")
        self.var_rr.set("")
        self.var_temp.set("")
        for v in self.symptoms.values():
            v.set("No")

    def on_select_row(self, _evt):
        sel = self.tree.selection()
        if not sel: return
        pid = self.tree.set(sel[0], "id")
        for p in self.queue.snapshot():
            if p.patient_id == pid:
                rules_ids = p.payload.get("rules", [])
                txt = explain_rules(rules_ids) if rules_ids else "—"
                self._set_rules_text(txt)
                self._set_badge(p.triage)
                return

    # ------- utility -------
    def _tri_to_01n(self, v):
        v = (v or "").strip().lower()
        if v in ("sì", "si", "y", "yes"): return 1
        if v in ("no", "n"):              return 0
        return None

    def _set_badge(self, triage):
        text = triage if triage in TRIAGE_COLORS else "—"
        bg = TRIAGE_COLORS.get(triage, "#bdc3c7")
        fg = "#000000"
        self.triage_badge.config(text=text, bg=bg, fg=fg)

    def _set_rules_text(self, txt):
        self.rules_text.configure(state="normal")
        self.rules_text.delete("1.0", "end")
        self.rules_text.insert("1.0", txt or "—")
        self.rules_text.configure(state="disabled")

    def _recount(self):
        counts = {k: 0 for k in SEVERITY_ORDER}
        for p in self.queue.snapshot():
            counts[p.triage] += 1
        return counts

    def _update_counters(self):
        counts = self._recount()
        # badge ordine: Rosso, Giallo, Verde, Bianco
        for name in ["Rosso", "Giallo", "Verde", "Bianco"]:
            f = self.badge_frames[name]
            f.count_lbl.config(text=str(counts.get(name, 0)))

    def _refresh_queue_table(self):
        # svuota
        for iid in self.tree.get_children():
            self.tree.delete(iid)
        snap = self.queue.snapshot()
        for i, p in enumerate(snap, start=1):
            name = p.payload.get("name", "")
            ts = datetime.utcfromtimestamp(p.arrival_ts).strftime("%H:%M:%S")
            self.tree.insert("", "end", values=(i, p.patient_id, name, p.triage, ts), tags=(p.triage,))

        self._update_counters()

if __name__ == "__main__":
    app = TriageApp()
    app.mainloop()
