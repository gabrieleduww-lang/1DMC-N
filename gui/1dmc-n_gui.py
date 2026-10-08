#!/usr/bin/env python3
"""Tk front end for the 1dmc-n engine. Needs tkinter, matplotlib and numpy.

Every run prints the exact engine command line in the console at the bottom,
so you can repeat it from a terminal.
"""
import json
import math
import os
import queue
import shutil
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

PALETTE = ["#6baed6", "#fd8d3c", "#74c476", "#9e9ac8", "#fdd049", "#e377c2", "#8c6d31", "#17becf"]
MATERIAL_COLORS = {"water": "#6baed6", "polyethylene": "#c7e9c0", "borated_pe_5": "#74c476",
                   "graphite": "#636363", "concrete": "#bdbdbd", "iron": "#8c6d62",
                   "lead": "#4a4a6a", "aluminum": "#d9d9d9"}


def find_engine():
    here = Path(__file__).resolve().parent
    names = ["1dmc-n.exe", "1dmc-n"] if os.name == "nt" else ["1dmc-n"]
    cands = [os.environ.get("ONEDMC_N_BIN", "")]
    for n in names:
        cands += [here.parent / "build" / n, here / n, here.parent / n]
    cands.append(shutil.which("1dmc-n") or "")
    for c in cands:
        if c and Path(c).is_file():
            return str(c)
    return None


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("1DMC-N - 1D Monte Carlo Neutron/Gamma Shielding")
        self.geometry("1280x820")
        self.engine = find_engine()
        self.q = queue.Queue()
        self.busy = False
        self.result = None
        self.custom = []  # strings like name:rho:H=..,O=..
        self.materials = ["water"]
        self._build_ui()
        self.after(100, self._poll)
        if not self.engine:
            self.log("Engine not found. Build it with `make` or choose the binary (Engine menu).")
        else:
            self.log(f"Engine: {self.engine}")
            self._load_materials()
        for m, t in [("polyethylene", 15), ("borated_pe_5", 5), ("lead", 3)]:
            self.tree.insert("", "end", values=(m, t))
        self.draw_geometry()

    def _build_ui(self):
        menu = tk.Menu(self)
        em = tk.Menu(menu, tearoff=0)
        em.add_command(label="Choose engine binary...", command=self.choose_engine)
        em.add_command(label="Quit", command=self.destroy)
        menu.add_cascade(label="Engine", menu=em)
        self.config(menu=menu)

        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True)
        left = ttk.Frame(paned, padding=6)
        right = ttk.PanedWindow(paned, orient="vertical")
        paned.add(left, weight=0)
        paned.add(right, weight=1)

        gf = ttk.LabelFrame(left, text="Geometry (source side first)")
        gf.pack(fill="x")
        self.tree = ttk.Treeview(gf, columns=("mat", "t"), show="headings", height=6, selectmode="browse")
        self.tree.heading("mat", text="Material")
        self.tree.heading("t", text="Thickness [cm]")
        self.tree.column("mat", width=130)
        self.tree.column("t", width=110, anchor="e")
        self.tree.pack(fill="x", padx=4, pady=4)
        self.tree.bind("<<TreeviewSelect>>", lambda e: self.draw_geometry())
        row = ttk.Frame(gf); row.pack(fill="x", padx=4)
        self.mat_cb = ttk.Combobox(row, values=self.materials, width=14, state="readonly")
        self.mat_cb.current(0); self.mat_cb.pack(side="left")
        self.thk = tk.StringVar(value="10")
        ttk.Entry(row, textvariable=self.thk, width=7).pack(side="left", padx=4)
        ttk.Button(row, text="Add", width=5, command=self.add_layer).pack(side="left")
        row2 = ttk.Frame(gf); row2.pack(fill="x", padx=4, pady=4)
        for txt, cmd in [("Remove", self.remove_layer), ("Up", lambda: self.move(-1)),
                         ("Down", lambda: self.move(1)), ("Edit t", self.edit_layer)]:
            ttk.Button(row2, text=txt, width=7, command=cmd).pack(side="left", padx=1)
        cf = ttk.Frame(gf); cf.pack(fill="x", padx=4, pady=(0, 4))
        self.custom_var = tk.StringVar(value="name:1.0:H=0.1,C=0.9")
        ttk.Entry(cf, textvariable=self.custom_var).pack(side="left", fill="x", expand=True)
        ttk.Button(cf, text="+ material", command=self.add_custom).pack(side="left", padx=2)

        sf = ttk.LabelFrame(left, text="Source"); sf.pack(fill="x", pady=6)
        self.src = tk.StringVar(value="cf252"); self.energy = tk.StringVar(value="2.0")
        self.angle = tk.StringVar(value="beam")
        self._row(sf, "Spectrum", lambda f: ttk.Combobox(f, textvariable=self.src, width=10, state="readonly",
                                               values=["mono", "cf252", "u235", "thermal", "gamma", "cs137", "co60"]))
        self._row(sf, "Energy [MeV] (mono/gamma)", lambda f: ttk.Entry(f, textvariable=self.energy, width=10))
        self._row(sf, "Incidence", lambda f: ttk.Combobox(f, textvariable=self.angle, width=10, state="readonly",
                                                values=["beam", "cosine"]))

        rf = ttk.LabelFrame(left, text="Run"); rf.pack(fill="x")
        self.nhist = tk.StringVar(value="500000"); self.batches = tk.StringVar(value="10")
        self.bins = tk.StringVar(value="60"); self.seed = tk.StringVar(value="12345")
        self.split = tk.StringVar(value="1.0"); self.analog = tk.BooleanVar(value=False)
        self._row(rf, "Histories", lambda f: ttk.Entry(f, textvariable=self.nhist, width=10))
        self._row(rf, "Batches", lambda f: ttk.Entry(f, textvariable=self.batches, width=10))
        self._row(rf, "Depth bins", lambda f: ttk.Entry(f, textvariable=self.bins, width=10))
        self._row(rf, "Seed", lambda f: ttk.Entry(f, textvariable=self.seed, width=10))
        self._row(rf, "Split ratio (1=off)", lambda f: ttk.Entry(f, textvariable=self.split, width=10))
        self.gamma = tk.BooleanVar(value=True); self.geom = tk.StringVar(value="AP")
        self._row(rf, "Dose geometry", lambda f: ttk.Combobox(f, textvariable=self.geom, width=10, state="readonly",
                                                              values=["AP", "PA", "LLAT", "RLAT", "ROT", "ISO"]))
        self.tsl = tk.BooleanVar(value=True); self.detailed = tk.BooleanVar(value=True)
        ttk.Checkbutton(rf, text="Secondary photons (capture / inelastic)", variable=self.gamma).pack(anchor="w", padx=6)
        ttk.Checkbutton(rf, text="Bound-H thermal scattering S(a,b)", variable=self.tsl).pack(anchor="w", padx=6)
        ttk.Checkbutton(rf, text="Measured gamma cascades (EGAF/RIPL)", variable=self.detailed).pack(anchor="w", padx=6)
        ttk.Checkbutton(rf, text="Analog capture (no survival biasing)", variable=self.analog).pack(anchor="w", padx=6)
        self.run_btn = ttk.Button(left, text="Run simulation", command=self.run_once)
        self.run_btn.pack(fill="x", pady=(8, 2))

        sc = ttk.LabelFrame(left, text="Thickness scan (selected layer)"); sc.pack(fill="x", pady=6)
        self.scan_max = tk.StringVar(value="60"); self.scan_pts = tk.StringVar(value="8")
        self._row(sc, "Max thickness [cm]", lambda f: ttk.Entry(f, textvariable=self.scan_max, width=10))
        self._row(sc, "Points", lambda f: ttk.Entry(f, textvariable=self.scan_pts, width=10))
        self.scan_btn = ttk.Button(sc, text="Run scan", command=self.run_scan)
        self.scan_btn.pack(fill="x", padx=6, pady=4)

        self.summary = tk.StringVar(value="No results yet")

        self.nb = ttk.Notebook(right)
        right.add(self.nb, weight=4)
        self.geo_canvas = tk.Canvas(self.nb, bg="white")
        self.geo_canvas.bind("<Configure>", lambda e: self.draw_geometry())
        self.nb.add(self.geo_canvas, text="Geometry")
        self.fig_flux, self.ax_flux = self._plot_tab("Flux vs depth")
        self.fig_spec, self.ax_spec = self._plot_tab("Exit spectrum")
        self.fig_scan, self.ax_scan = self._plot_tab("Thickness scan")
        self.res_text = tk.Text(self.nb, font=("TkFixedFont", 10), state="disabled", wrap="none")
        self.nb.add(self.res_text, text="Results")

        lf = ttk.Frame(right); right.add(lf, weight=1)
        self.console = tk.Text(lf, height=9, bg="#101418", fg="#c9d1d9", insertbackground="white",
                               font=("TkFixedFont", 9), state="disabled")
        sb = ttk.Scrollbar(lf, command=self.console.yview)
        self.console.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y"); self.console.pack(fill="both", expand=True)

    def _row(self, parent, label, make_widget):
        f = ttk.Frame(parent); f.pack(fill="x", padx=6, pady=1)
        ttk.Label(f, text=label, width=20).pack(side="left")
        make_widget(f).pack(side="left")

    def _plot_tab(self, title):
        frame = ttk.Frame(self.nb); self.nb.add(frame, text=title)
        fig = Figure(figsize=(6, 4), dpi=100, tight_layout=True)
        ax = fig.add_subplot(111)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        NavigationToolbar2Tk(canvas, frame).update()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        fig._tk_canvas = canvas
        return fig, ax

    def log(self, msg):
        self.console.configure(state="normal")
        self.console.insert("end", msg + "\n"); self.console.see("end")
        self.console.configure(state="disabled")

    def choose_engine(self):
        p = filedialog.askopenfilename(title="Select 1dmc-n binary")
        if p:
            self.engine = p; self.log(f"Engine: {p}"); self._load_materials()

    def _load_materials(self):
        try:
            out = subprocess.run([self.engine, "--list-materials"] + self._custom_args(),
                                 capture_output=True, text=True, check=True).stdout
            self.materials = [m["name"] for m in json.loads(out)]
            self.mat_cb.configure(values=self.materials)
            if self.mat_cb.get() not in self.materials:
                self.mat_cb.current(0)
        except Exception as e:
            self.log(f"Could not list materials: {e}")

    def _custom_args(self):
        a = []
        for c in self.custom:
            a += ["--material", c]
        return a

    def add_custom(self):
        s = self.custom_var.get().strip()
        if s.count(":") != 2:
            messagebox.showerror("Custom material", "Format: name:density:Nuc=massfrac,Nuc=massfrac\n"
                                 "Nuclides: H C O Na Al Si Ca Fe Pb B10 B11")
            return
        self.custom.append(s); self._load_materials(); self.log(f"Added material {s.split(':')[0]}")

    def layers(self):
        return [(self.tree.item(i, "values")[0], float(self.tree.item(i, "values")[1]))
                for i in self.tree.get_children()]

    def add_layer(self):
        try:
            t = float(self.thk.get()); assert t > 0
        except Exception:
            return messagebox.showerror("Layer", "Thickness must be a positive number")
        self.tree.insert("", "end", values=(self.mat_cb.get(), t)); self.draw_geometry()

    def remove_layer(self):
        for s in self.tree.selection():
            self.tree.delete(s)
        self.draw_geometry()

    def move(self, d):
        for s in self.tree.selection():
            self.tree.move(s, "", self.tree.index(s) + d)
        self.draw_geometry()

    def edit_layer(self):
        sel = self.tree.selection()
        if not sel:
            return
        v = self.tree.item(sel[0], "values")
        try:
            t = float(self.thk.get()); assert t > 0
        except Exception:
            return messagebox.showerror("Layer", "Enter a positive thickness in the box next to 'Add'")
        self.tree.item(sel[0], values=(self.mat_cb.get() or v[0], t)); self.draw_geometry()

    def draw_geometry(self):
        c = self.geo_canvas; c.delete("all")
        L = self.layers() if hasattr(self, "tree") else []
        W, H = max(c.winfo_width(), 400), max(c.winfo_height(), 200)
        if not L:
            c.create_text(W / 2, H / 2, text="Add layers to define the shield"); return
        tot = sum(t for _, t in L); x0, x1, y0, y1 = 90, W - 90, H * 0.28, H * 0.72
        sel = self.tree.selection()
        sel_idx = self.tree.index(sel[0]) if sel else -1
        x = x0
        for i, (m, t) in enumerate(L):
            w = (x1 - x0) * t / tot
            c.create_rectangle(x, y0, x + w, y1, fill=MATERIAL_COLORS.get(m, PALETTE[i % 8]),
                               outline="red" if i == sel_idx else "black", width=3 if i == sel_idx else 1)
            c.create_text(x + w / 2, y0 - 14, text=f"{m}", font=("TkDefaultFont", 9, "bold"))
            c.create_text(x + w / 2, y1 + 14, text=f"{t:g} cm")
            x += w
        c.create_line(20, (y0 + y1) / 2, x0 - 4, (y0 + y1) / 2, arrow="last", width=3, fill="#d62728")
        c.create_text(55, (y0 + y1) / 2 - 14, text="n source", fill="#d62728")
        c.create_line(x1 + 4, (y0 + y1) / 2, W - 20, (y0 + y1) / 2, arrow="last", width=2, dash=(4, 3))
        c.create_text(W - 55, (y0 + y1) / 2 - 14, text="transmitted")
        c.create_text(W / 2, H - 20, text=f"Total thickness {tot:g} cm   (1D slab, infinite in y and z)",
                      fill="#555")

    def _args(self, layers, n=None, bins=None):
        a = [self.engine]
        for m, t in layers:
            a += ["--layer", f"{m}:{t}"]
        a += self._custom_args()
        a += ["--source", self.src.get(), "--energy", self.energy.get(), "--angle", self.angle.get(),
              "--n", str(n or self.nhist.get()), "--batches", self.batches.get(),
              "--bins", str(bins or self.bins.get()), "--seed", self.seed.get(), "--split", self.split.get()]
        a += ["--dose", self.geom.get()]
        if self.analog.get():
            a.append("--analog")
        if not self.gamma.get():
            a.append("--no-gamma")
        if not self.tsl.get():
            a.append("--no-tsl")
        if not self.detailed.get():
            a.append("--generic-gamma")
        return a

    def _check(self):
        if self.busy:
            return False
        if not self.engine:
            messagebox.showerror("Engine", "1dmc-n binary not found. Build it first (make).")
            return False
        if not self.layers():
            messagebox.showerror("Geometry", "Add at least one layer."); return False
        return True

    def _worker(self, jobs, kind):
        try:
            out = []
            for tag, args in jobs:
                self.q.put(("log", "$ " + " ".join(args)))
                r = subprocess.run(args, capture_output=True, text=True)
                if r.returncode != 0:
                    raise RuntimeError(r.stderr.strip() or "engine failed")
                out.append((tag, json.loads(r.stdout)))
                self.q.put(("log", f"  done ({tag}) in {out[-1][1]['runtime_s']:.2f}s"))
            self.q.put((kind, out))
        except Exception as e:
            self.q.put(("error", str(e)))

    def _start(self, jobs, kind):
        self.busy = True; self.run_btn.state(["disabled"]); self.scan_btn.state(["disabled"])
        threading.Thread(target=self._worker, args=(jobs, kind), daemon=True).start()

    def run_once(self):
        if self._check():
            self._start([("run", self._args(self.layers()))], "run")

    def run_scan(self):
        if not self._check():
            return
        sel = self.tree.selection()
        L = self.layers()
        idx = self.tree.index(sel[0]) if sel else len(L) - 1
        try:
            tmax, npts = float(self.scan_max.get()), int(self.scan_pts.get()); assert tmax > 0 and npts >= 2
        except Exception:
            return messagebox.showerror("Scan", "Invalid scan parameters")
        jobs = []
        for t in np.linspace(tmax / npts, tmax, npts):
            LL = list(L); LL[idx] = (L[idx][0], float(t))
            jobs.append((float(t), self._args(LL, bins=10)))
        self.scan_label = f"{L[idx][0]} (layer {idx + 1})"
        self._start(jobs, "scan")

    def _poll(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self.log(payload)
                elif kind == "error":
                    self.busy = False; self.run_btn.state(["!disabled"]); self.scan_btn.state(["!disabled"])
                    self.log("ERROR: " + payload); messagebox.showerror("Engine error", payload)
                else:
                    self.busy = False; self.run_btn.state(["!disabled"]); self.scan_btn.state(["!disabled"])
                    (self.show_run if kind == "run" else self.show_scan)(payload)
        except queue.Empty:
            pass
        self.after(100, self._poll)

    @staticmethod
    def _fmt(s):
        return f"{s['mean']:.3e} +/- {s['stderr']:.1e}"

    def show_run(self, out):
        d = out[0][1]; self.result = d
        n, g, ds = d["neutron"], d["photon"], d["dose"]
        psrc = d["source"]["particle"] == "photon"
        txt = ""
        if not psrc:
            txt += (f"NEUTRONS  T {self._fmt(n['transmitted'])}\n"
                    f"          R {self._fmt(n['reflected'])}\n"
                    f"          A {self._fmt(n['absorbed'])}\n"
                    f"  uncollided {self._fmt(n['uncollided_transmitted'])}\n")
        txt += (f"PHOTONS   T {self._fmt(g['transmitted'])}\n"
                f"          R {self._fmt(g['reflected'])}\n")
        if not psrc:
            txt += (f"  made: {g['produced_from_capture']:.3f}/n capture, "
                    f"{g['produced_from_inelastic']:.3f}/n inelastic\n")
        txt += (f"(analytic uncollided {d['uncollided_analytic']:.3e})\n"
                f"DOSE ratio  total {self._fmt(ds['total_ratio'])}\n"
                f"   neutron {ds['neutron_ratio']['mean']:.3e}  photon {ds['photon_ratio']['mean']:.3e}\n"
                f"{d['runtime_s']:.2f}s on {d['threads']} thread(s)")
        phys = d.get("physics", {})
        notes = "\n".join(["Thermal: " + x for x in phys.get("thermal", [])] or ["Thermal: free-gas for all nuclides"])
        notes += f"\nGamma model: {phys.get('gamma_model', '?')}"
        if not psrc and g.get("capture_events", 0) > 0:
            notes += f"\nMeasured (EGAF) cascade used for {100 * g['capture_measured_cascade_fraction']:.0f}% of captures"
        self.summary.set(txt); self.log(txt)
        self.res_text.configure(state="normal"); self.res_text.delete("1.0", "end")
        self.res_text.insert("end", txt + "\n\n" + notes + "\n\nT = transmitted, R = reflected, A = absorbed (per source particle, mean +/- std. error).\n"
                             "Dose ratio = exit-plane dose / incident dose, ICRP-116 effective dose, selected geometry.\n"
                             "Neutron results are 1D, continuous-energy ENDF/B-VIII.0; see README for limits.")
        self.res_text.configure(state="disabled")

        edges = np.array(d["depth_edges_cm"]); xc = 0.5 * (edges[1:] + edges[:-1])
        fl = {k: np.array(v) for k, v in d["flux_depth"].items()}
        ax = self.ax_flux; ax.clear()
        ntot = fl["neutron_fast"] + fl["neutron_intermediate"] + fl["neutron_thermal"]
        series = [("neutron total", ntot, "black", 2), ("fast (>100 keV)", fl["neutron_fast"], "#d62728", 1),
                  ("intermediate", fl["neutron_intermediate"], "#2ca02c", 1),
                  ("thermal", fl["neutron_thermal"], "#1f77b4", 1), ("photon", fl["photon"], "#ff7f0e", 2)]
        for lab, v, col, lw in series:
            if np.any(v > 0):
                ax.semilogy(xc, np.where(v > 0, v, np.nan), color=col, lw=lw, label=lab)
        for l in d["layers"][1:]:
            ax.axvline(l["x0"], color="gray", ls="--", lw=0.8)
        ax.set_xlabel("depth [cm]"); ax.set_ylabel("fluence per source particle [cm$^{-2}$]")
        ax.set_title("Flux profile (track-length estimator)"); ax.grid(alpha=0.3); ax.legend()
        self.fig_flux._tk_canvas.draw()

        ax = self.ax_spec; ax.clear()
        for pre, lab, style in [("neutron", "n", "-"), ("photon", "\u03b3", "--")]:
            se = np.array(d[f"{pre}_spectrum_edges_eV"]); xm = np.sqrt(se[1:] * se[:-1]); du = np.log(se[1:] / se[:-1])
            for key, what in [("transmitted", "transmitted"), ("reflected", "reflected")] + \
                             ([("produced", "produced")] if pre == "photon" else []):
                if f"{pre}_spectrum_{key}" not in d:
                    continue
                v = np.array(d[f"{pre}_spectrum_{key}"]) / du
                if np.any(v > 0):
                    ax.loglog(xm, np.where(v > 0, v, np.nan), style, drawstyle="steps-mid", label=f"{lab} {what}")
        ax.set_xlabel("energy [eV]"); ax.set_ylabel("particles per unit lethargy per source particle")
        ax.set_title("Exit / production spectra"); ax.grid(alpha=0.3, which="both"); ax.legend()
        self.fig_spec._tk_canvas.draw()
        self.nb.select(1)

    def show_scan(self, out):
        t = np.array([o[0] for o in out])
        def col(*path):
            def get(o):
                x = o[1]
                for p in path:
                    x = x[p]
                return x
            return np.array([get(o)["mean"] for o in out]), np.array([get(o)["stderr"] for o in out])
        Dt, Dte = col("dose", "total_ratio"); Dn, _ = col("dose", "neutron_ratio"); Dg, _ = col("dose", "photon_ratio")
        ax = self.ax_scan; ax.clear()
        for v, e, fmt, lab in [(Dt, Dte, "o-", "total dose"), (Dn, None, "s--", "neutron dose"), (Dg, None, "^--", "photon dose")]:
            ok = v > 0
            if ok.any():
                ax.errorbar(t[ok], v[ok], None if e is None else e[ok], fmt=fmt, label=lab)
        ax.set_yscale("log"); ax.set_xlabel(f"thickness of {self.scan_label} [cm]")
        ax.set_ylabel("dose transmission ratio"); ax.grid(alpha=0.3, which="both"); ax.legend()
        ax.set_title("Attenuation vs thickness")
        msgs = []
        for name, v in [("total", Dt), ("neutron", Dn)]:
            m = (v > 0) & (t >= t.max() / 2)
            if m.sum() >= 2:
                sl = np.polyfit(t[m], np.log(v[m]), 1)[0]
                if sl < 0:
                    msgs.append(f"{name} dose: HVL ~ {math.log(2) / -sl:.1f} cm, TVL ~ {math.log(10) / -sl:.1f} cm")
        if msgs:
            ax.text(0.03, 0.05, "\n".join(msgs), transform=ax.transAxes); self.log("Scan (tail fit): " + "; ".join(msgs))
        self.fig_scan._tk_canvas.draw(); self.nb.select(3)


if __name__ == "__main__":
    App().mainloop()
