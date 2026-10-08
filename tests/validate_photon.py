#!/usr/bin/env python3
"""Check data/photon/*.csv against NIST XCOM mass attenuation coefficients (with coherent).

Anchor values: NIST XCOM / Hubbell & Seltzer, cm^2/g. Tolerance 3 %.
"""
import csv, sys
from pathlib import Path
import numpy as np

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "data")
NA = 0.602214076
ANCHORS = {   # material: (mass fractions by element, {E_MeV: mu/rho})
    "H":     ({"H": 1.0}, {1.0: 0.1263}),
    "water": ({"H": 0.111894, "O": 0.888106},
              {0.1: 0.1707, 0.5: 0.09687, 1.0: 0.07072, 2.0: 0.04942, 5.0: 0.03030, 10.0: 0.02219}),
    "iron":  ({"Fe": 1.0}, {0.1: 0.3717, 0.5: 0.08414, 1.0: 0.05995, 2.0: 0.04250, 5.0: 0.03106, 10.0: 0.02995}),
    "lead":  ({"Pb": 1.0}, {0.1: 5.549, 0.5: 0.1614, 1.0: 0.07102, 2.0: 0.04606, 3.0: 0.04234,
                            5.0: 0.04272, 10.0: 0.04972}),
}

def load(sym):
    A = None; rows = []
    for line in open(DATA / "photon" / f"{sym}.csv"):
        if line.startswith("#atomic_weight"): A = float(line.split()[1])
        elif line[0] in "#E": continue
        else: rows.append([float(x) for x in line.split(",")])
    r = np.array(rows); return A, r

def mu_rho(frac, E_MeV):
    tot = 0.0
    for sym, w in frac.items():
        A, r = load(sym)
        e = r[:, 0] * 1e-6
        s = sum(np.exp(np.interp(np.log(E_MeV), np.log(e), np.log(np.maximum(r[:, k], 1e-30)))) for k in (1, 2, 3, 4))
        tot += w * s * NA / A
    return tot

worst = 0
for name, (frac, pts) in ANCHORS.items():
    for E, ref in pts.items():
        v = mu_rho(frac, E); d = 100 * (v / ref - 1); worst = max(worst, abs(d))
        print(f"  {name:6s} {E:5.1f} MeV  data {v:.5f}  NIST {ref:.5f}  {d:+.1f}%")
print(f"worst deviation {worst:.1f}%")
sys.exit(0 if worst < 3.0 else 1)
