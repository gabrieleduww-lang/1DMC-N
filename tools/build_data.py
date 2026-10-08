#!/usr/bin/env python3
"""Build the element/isotope tables, photon cross sections and ICRP-116 dose tables for 1DMC-N.

Neutron data, gamma cascades and thermal-scattering tables are built by their own scripts; run
tools/fetch_and_build.sh to do everything in the right order.

  * Isotope masses/abundances : the `periodictable` Python package
  * Photon cross sections : Elam tables (`xraydb`) up to 0.5 MeV; above that Klein-Nishina (incoherent),
        a Geant4-style Bethe-Heitler fit (pair production) and a fitted photoelectric extrapolation.
        Validated against NIST XCOM values by tests/validate_photon.py
  * Dose coefficients : ICRP-116 effective dose per fluence as bundled with OpenMC
        (pass --icrp-dir with neutrons.txt / photons.txt)

Usage
  python3 tools/build_data.py --icrp-dir icrp116 --out data
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------ settings
NB_P = 600                    # photon energy bins (log-uniform)
EMIN_P, EMAX_P = 1e4, 2e7     # eV
ELEMENTS = ["H", "Li", "B", "C", "N", "O", "Na", "Mg", "Al", "Si", "K", "Ca",
            "Cr", "Mn", "Fe", "Ni", "Cu", "Cd", "Gd", "W", "Pb"]
MIN_ABUNDANCE = 0.1           # percent; lighter isotopes are dropped and abundances renormalised
NEUTRON_AMU = 1.00866491588
AMU_MEV = 931.49410242
NA = 0.602214076              # atoms/(barn cm) per (mol/g)... i.e. N_A * 1e-24


# ------------------------------------------------------------------ photon data
ME = 0.51099895  # MeV
R2 = 0.07940787  # r_e^2 in barn


def klein_nishina(E_mev):
    k = E_mev / ME
    return 2 * math.pi * R2 * ((1 + k) / k**2 * (2 * (1 + k) / (1 + 2 * k) - math.log(1 + 2 * k) / k)
                               + math.log(1 + 2 * k) / (2 * k) - (1 + 3 * k) / (1 + 2 * k) ** 2)


def bethe_heitler(Z, E_mev):
    """Pair production per atom (barn), Geant4-style parametrisation (valid 1.5 MeV-100 GeV)."""
    a = [8.7842e2, -1.9625e3, 1.2949e3, -2.0028e2, 1.2575e1, -2.8333e-1]
    b = [-1.0342e1, 1.7692e1, -8.2381, 1.3063, -9.0815e-2, 2.3586e-3]
    c = [-4.5263e2, 1.1161e3, -8.6749e2, 2.1773e2, -2.0467e1, 6.5372e-1]
    if E_mev <= 2 * ME:
        return 0.0
    X = math.log(max(E_mev, 1.5) / ME)
    F = lambda k: sum(k[i] * X**i for i in range(6))
    s = (Z + 1) * Z * (F(a) + F(b) * Z + F(c) / Z) * 1e-6
    if E_mev < 1.5:
        s *= ((E_mev - 2 * ME) / (1.5 - 2 * ME)) ** 2
    return max(s, 0.0)


def photon_table(sym):
    import xraydb
    import warnings
    warnings.filterwarnings("ignore")
    Z = xraydb.atomic_number(sym)
    Aw = xraydb.atomic_mass(sym)
    conv = Aw / NA                      # cm2/g -> barn/atom
    E = np.logspace(math.log10(EMIN_P), math.log10(EMAX_P), NB_P + 1)
    Ec = np.sqrt(E[1:] * E[:-1])
    E0 = 5e5                            # eV, last energy taken from Elam tables
    pe0 = xraydb.mu_elam(sym, E0, kind="photo") * conv
    inc0 = xraydb.mu_elam(sym, E0, kind="incoh") * conv
    coh0 = xraydb.mu_elam(sym, E0, kind="coh") * conv
    S0 = inc0 / (klein_nishina(E0 * 1e-6) * Z)
    rows = []
    for e in Ec:
        em = e * 1e-6
        if e <= E0:
            pe = xraydb.mu_elam(sym, e, kind="photo") * conv
            inc = xraydb.mu_elam(sym, e, kind="incoh") * conv
            coh = xraydb.mu_elam(sym, e, kind="coh") * conv
        else:
            S = 1 - (1 - S0) * (E0 / e)
            inc = klein_nishina(em) * Z * S
            coh = coh0 * (E0 / e) ** 2
            # photoelectric above Elam range: three-segment power law scaled from the 0.5 MeV Elam value
            # (matters only for high Z; <2% of mu/rho above 1 MeV for Z < 30)
            x1, x2 = 1e6, 3e6
            s1, s2, s3 = 2.20, 1.75, 2.30   # empirical, fitted to NIST lead 1-10 MeV (<1%)
            if e <= x1:
                pe = pe0 * (e / E0) ** -s1
            elif e <= x2:
                pe = pe0 * (x1 / E0) ** -s1 * (e / x1) ** -s2
            else:
                pe = pe0 * (x1 / E0) ** -s1 * (x2 / x1) ** -s2 * (e / x2) ** -s3
        rows.append((e, pe, inc, coh, bethe_heitler(Z, em)))
    return Z, Aw, rows


def write_photon(sym, outdir):
    Z, Aw, rows = photon_table(sym)
    p = outdir / "photon" / f"{sym}.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        f.write(f"# 1DMC-N photon cross sections for {sym} (Z={Z}), barn/atom\n")
        f.write(f"#grid {NB_P} {EMIN_P:g} {EMAX_P:g}\n#Z {Z}\n#atomic_weight {Aw:.5f}\nE_eV,photo,incoh,coh,pair\n")
        for r in rows:
            f.write(",".join("0" if v == 0 else f"{v:.5g}" for v in r) + "\n")
    return Z, Aw


# ------------------------------------------------------------------ dose tables
def convert_dose(src: Path, dst: Path, emax_mev):
    rows = []
    for line in src.read_text(errors="ignore").splitlines():
        t = line.split()
        if len(t) >= 7:
            try:
                v = [float(x) for x in t[:7]]
            except ValueError:
                continue
            if v[0] <= emax_mev:
                rows.append(v)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as f:
        f.write("# ICRP-116 effective dose per fluence, pSv cm^2 (values as bundled with OpenMC, MIT licence)\n")
        f.write("E_MeV,AP,PA,LLAT,RLAT,ROT,ISO\n")
        for v in rows:
            f.write(",".join(f"{x:.6g}" for x in v) + "\n")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--icrp-dir", required=True, help="dir with ICRP-116 neutrons.txt / photons.txt")
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import periodictable as pt

    # isotope list
    iso_rows, elem_rows = [], []
    for sym in ELEMENTS:
        el = getattr(pt.elements, sym)
        isos = [(i, el[i].abundance, el[i].mass) for i in el.isotopes if el[i].abundance >= MIN_ABUNDANCE]
        tot = sum(x[1] for x in isos)
        for i, ab, m in isos:
            nxt = el[i + 1].mass if (i + 1) in el.isotopes else None
            sn = (m + NEUTRON_AMU - nxt) * AMU_MEV if nxt else 7.5
            iso_rows.append((sym, i, ab / tot, m, sn))
    print("isotopes:", len(iso_rows))

    # photon data
    for sym in ELEMENTS:
        Z, Aw = write_photon(sym, out)
        elem_rows.append((sym, Z, Aw))
    with open(out / "elements.csv", "w") as f:
        f.write("symbol,Z,atomic_weight\n")
        for r in elem_rows:
            f.write(f"{r[0]},{r[1]},{r[2]:.5f}\n")
    with open(out / "isotopes.csv", "w") as f:
        f.write("symbol,mass_number,abundance,atomic_mass_amu,Sn_MeV\n")
        for r in iso_rows:
            f.write(f"{r[0]},{r[1]},{r[2]:.6f},{r[3]:.6f},{r[4]:.4f}\n")

    # dose tables
    d = Path(a.icrp_dir)
    convert_dose(d / "neutrons.txt", out / "dose" / "icrp116_neutron.csv", 20.0)
    convert_dose(d / "photons.txt", out / "dose" / "icrp116_photon.csv", 20.0)

    print("done")


if __name__ == "__main__":
    sys.exit(main())
