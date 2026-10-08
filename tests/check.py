#!/usr/bin/env python3
"""Self-consistency and data checks (run: make check).

These are NOT a substitute for benchmarking against MCNP/OpenMC/Geant4 or measured data.
"""
import json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = os.environ.get("BIN", str(ROOT / "build" / "1dmc-n"))
fails = []

def run(*args):
    r = subprocess.run([BIN, *args], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"engine failed: {r.stderr}")
    return json.loads(r.stdout)

def check(name, ok, msg):
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {msg}")
    if not ok:
        fails.append(name)

def agree(a, b):  # two {"mean","stderr"} dicts
    return abs(a["mean"] - b["mean"]) < 4 * (a["stderr"] ** 2 + b["stderr"] ** 2) ** 0.5

# 1 neutron uncollided vs analytic
d = run("--layer", "lead:5", "--energy", "2", "--n", "400000"); u = d["neutron"]["uncollided_transmitted"]
check("neutron uncollided", abs(u["mean"] - d["uncollided_analytic"]) < 5 * u["stderr"],
      f"MC {u['mean']:.4e}+/-{u['stderr']:.1e} vs analytic {d['uncollided_analytic']:.4e}")

# 2 photon uncollided vs analytic
d = run("--layer", "water:10", "--source", "co60", "--n", "400000"); u = d["photon"]["uncollided_transmitted"]
check("photon uncollided", abs(u["mean"] - d["uncollided_analytic"]) < 5 * u["stderr"],
      f"MC {u['mean']:.4e}+/-{u['stderr']:.1e} vs analytic {d['uncollided_analytic']:.4e}")

# 3 particle balance incl. (n,2n) and pair production
d = run("--layer", "lead:20", "--energy", "14", "--n", "100000"); n, g = d["neutron"], d["photon"]
sn = sum(n[k]["mean"] for k in ("transmitted", "reflected", "absorbed"))
sg = sum(g[k]["mean"] for k in ("transmitted", "reflected", "absorbed"))
pg = g["produced_from_capture"] + g["produced_from_inelastic"] + g["pair_extra"]
check("particle balance", abs(sn - 1 - n["multiplication_extra"]) < 1e-4 and abs(sg - pg) < 1e-4,
      f"neutrons {sn:.5f} vs {1 + n['multiplication_extra']:.5f}; photons {sg:.5f} vs {pg:.5f}")

# 4 hydrogen capture line
d = run("--layer", "water:30", "--source", "thermal", "--n", "60000"); g = d["photon"]
e = g["produced_energy_MeV_capture"] / g["produced_from_capture"]
check("H(n,g) line", 2.15 < e < 2.24, f"mean capture-photon energy in water {e:.4f} MeV")

# 5 implicit vs analog capture
a = run("--layer", "water:10", "--energy", "1", "--n", "150000", "--analog")["neutron"]["transmitted"]
b = run("--layer", "water:10", "--energy", "1", "--n", "150000")["neutron"]["transmitted"]
check("implicit vs analog", agree(a, b), f"{a['mean']:.4e} vs {b['mean']:.4e}")

# 6 splitting
a = run("--layer", "concrete:40", "--source", "cf252", "--n", "150000")["dose"]["total_ratio"]
b = run("--layer", "concrete:40", "--source", "cf252", "--n", "150000", "--split", "1.3")["dose"]["total_ratio"]
check("splitting", agree(a, b), f"plain {a['mean']:.4e}+/-{a['stderr']:.1e} split {b['mean']:.4e}+/-{b['stderr']:.1e}")

# 8 thermal diffusion: asymptotic relaxation length of thermal neutrons in water (measured diffusion length ~2.7-2.8 cm)
import numpy as np
def relax_len(mat, *extra):
    d = run("--layer", f"{mat}:40", "--source", "thermal", "--angle", "cosine", "--n", "150000", "--bins", "60", *extra)
    x = np.array(d["depth_edges_cm"]); xc = 0.5 * (x[1:] + x[:-1]); f = np.array(d["flux_depth"]["neutron_thermal"])
    m = (xc > 6) & (xc < 16) & (f > 0)
    return -1.0 / np.polyfit(xc[m], np.log(f[m]), 1)[0]
L_sab, L_fg = relax_len("water"), relax_len("water", "--no-tsl")
check("water thermal diffusion (S(a,b))", 2.55 < L_sab < 3.05, f"relaxation length {L_sab:.2f} cm (measured diffusion length ~2.7-2.8 cm)")
check("S(a,b) matters", L_fg > 1.15 * L_sab, f"free-gas hydrogen gives {L_fg:.2f} cm vs {L_sab:.2f} cm with S(a,b)")

# 9 resonance structure kept: iron 24 keV window, uncollided = analytic pointwise value
d = run("--layer", "iron:20", "--energy", "0.024", "--n", "200000"); u = d["neutron"]["uncollided_transmitted"]
check("iron 24 keV window", abs(u["mean"] - d["uncollided_analytic"]) < 5 * u["stderr"] and d["uncollided_analytic"] > 0.3,
      f"uncollided {u['mean']:.4f} vs analytic {d['uncollided_analytic']:.4f} (window open)")

# 10 RIPL inelastic cascade: pure Fe-56 at 1.2 MeV only opens the 846.8 keV level
d = run("--material", "fe56:7.874:Fe56=1", "--layer", "fe56:10", "--energy", "1.2", "--n", "150000"); g = d["photon"]
e = g["produced_energy_MeV_inelastic"] / g["produced_from_inelastic"]
check("Fe-56 846.8 keV line", abs(e - 0.8468) < 0.002, f"mean inelastic photon energy {e:.4f} MeV")

# 11 EGAF cascades conserve the neutron separation energy per capture
for iso, sn in (("Fe56", 7.646), ("Pb207", 7.367), ("Si28", 8.474)):
    d = run("--material", f"x:3:{iso}=1", "--layer", "x:20", "--source", "thermal", "--n", "60000"); g = d["photon"]
    e = g["produced_energy_MeV_capture"] / g["capture_events"]
    check(f"{iso} capture energy", abs(e - sn) < 0.08, f"{e:.3f} MeV per capture (Sn {sn})")

# 7 photon tables vs NIST
r = subprocess.run([sys.executable, str(ROOT / "tests" / "validate_photon.py")], capture_output=True, text=True)
check("photon data vs NIST", r.returncode == 0, r.stdout.strip().splitlines()[-1])

print("\nAll checks passed." if not fails else f"\nFAILED: {fails}")
sys.exit(1 if fails else 0)
