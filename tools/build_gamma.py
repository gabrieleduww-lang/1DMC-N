#!/usr/bin/env python3
"""Build gamma-cascade tables for 1DMC-N.

capture cascades   <- EGAF (Evaluated Gamma-ray Activation File, IAEA), thermal-capture
                      level schemes with absolute gamma intensities. JSON copy used here:
                      https://github.com/AaronMHurst/python_egaf  (pyEGAF/EGAF_JSON)
level schemes      <- RIPL-3 discrete levels as distributed with TALYS:
                      https://github.com/arjankoning1/talys  (structure/levels/exp/*.lev)

Outputs (under <out>/gamma):
  capture/<iso>.csv   L,idx,E_keV  /  T,from,to,E_keV,P(per capture)   + '#' header with coverage info
  levels/<iso>.csv    L,idx,E_MeV  /  B,idx,final,branch_ratio,alpha_ICC

Usage: python3 tools/build_gamma.py --egaf <pyEGAF/EGAF_JSON> --levels <talys/structure/levels/exp> --out data
"""
import argparse
import csv
import json
import re
from pathlib import Path

ELEMENT_FILE_RE = re.compile(r"\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+.*?([A-Z][a-z]?)(\d{3})\s*$")


def parse_lev_file(path):
    """Return {(Z, A): [levels]} with level = dict(idx, E, branches=[(final, br, icc)])."""
    out, cur, lines = {}, None, Path(path).read_text(errors="ignore").splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        m = ELEMENT_FILE_RE.match(ln)
        if m and len(ln.split()) >= 5 and ln.split()[0].isdigit() and ln.split()[1].isdigit():
            z, a = int(m.group(1)), int(m.group(2))
            cur = out.setdefault((z, a), [])
            i += 1
            continue
        t = ln.split()
        if cur is not None and len(t) >= 5:
            try:
                idx, e, spin, par, nbr = int(t[0]), float(t[1]), float(t[2]), int(t[3]), int(t[4])
            except ValueError:
                i += 1
                continue
            br = []
            for k in range(nbr):
                bt = lines[i + 1 + k].split()
                br.append((int(bt[0]), float(bt[1]), float(bt[2])))
            cur.append({"idx": idx, "E": e, "branches": br})
            i += 1 + nbr
            continue
        i += 1
    return out


def write_levels(levels, path, emax=8.0):
    """Keep ground state, every level with branches below emax, and everything they feed."""
    keep = {L["idx"] for L in levels if L["idx"] == 0 or (L["branches"] and L["E"] <= emax)}
    byidx = {L["idx"]: L for L in levels}
    changed = True
    while changed:
        changed = False
        for k in list(keep):
            for f, _, _ in byidx[k]["branches"]:
                if f not in keep and f in byidx:
                    keep.add(f); changed = True
    with open(path, "w") as f:
        f.write("# RIPL-3 discrete levels via TALYS structure/levels/exp. L,idx,E_MeV ; B,idx,final,branch_ratio,alpha_ICC\n")
        for L in sorted((byidx[k] for k in keep), key=lambda x: x["idx"]):
            f.write(f"L,{L['idx']},{L['E']:.6f}\n")
            for fin, brr, icc in L["branches"]:
                f.write(f"B,{L['idx']},{fin},{brr:.6f},{icc:.5g}\n")
    return len(keep)


def build_capture(egaf_dir, target, out_path):
    fn = list(Path(egaf_dir).glob(f"j_EGAF_{target}_NG_*.json"))
    if not fn:
        return None
    d = json.load(open(fn[0]))
    d = d[0] if isinstance(d, list) else d
    sn_kev = d["recordQ"][0]["energyNeutronSeparationAME2020"]
    sigma = d["neutronCaptureNormalization"][0]["normalizationRecord"][0]["adoptedTotalThermalCaptureCrossSection"]
    levels = d["levelScheme"]
    cap = max(levels, key=lambda L: L["levelEnergy"])        # capture state = highest level (E ~ Sn)
    trans = []
    for L in levels:
        for g in L["gammaDecay"]:
            if not g["gammaAbsoluteIntensities"]:
                continue
            p = g["gammaAbsoluteIntensities"][0]["populationPerNeutronCapture"]
            if p and p > 0:
                trans.append((g["levelIndexInitial"], g["levelIndexFinal"], g["gammaEnergy"], p, g["gammaTransitionType"]))
    prim = sum(t[3] for t in trans if t[4] == "primary")
    e_tot = sum(t[2] * t[3] for t in trans)
    if not (0.05 <= prim <= 1.5) or abs(cap["levelEnergy"] - sn_kev) > 5.0:
        return {"target": target, "status": "rejected", "prim": prim}
    scale = 1.0 / prim if prim > 1.0 else 1.0     # primary populations cannot exceed 1 per capture
    with open(out_path, "w") as f:
        f.write(f"# EGAF thermal-capture cascade for {target}(n,g){d['nucleusID']}; Sn_keV={sn_kev:.3f} "
                f"sigma_th_b={sigma:g} primary_coverage={min(prim, 1.0):.4f} Egamma_over_Sn={e_tot / sn_kev:.3f}\n")
        f.write(f"S,{sn_kev:.4f},{cap['levelIndex']}\n")
        for L in levels:
            f.write(f"L,{L['levelIndex']},{L['levelEnergy']:.4f}\n")
        for a, b, e, p, kind in trans:
            f.write(f"T,{a},{b},{e:.4f},{p * (scale if kind == 'primary' else 1.0):.6g}\n")
    return {"target": target, "status": "ok", "prim": prim, "E/Sn": e_tot / sn_kev, "ntrans": len(trans)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--egaf", required=True)
    ap.add_argument("--levels", required=True, help="TALYS structure/levels/exp directory")
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    out = Path(a.out) / "gamma"
    (out / "capture").mkdir(parents=True, exist_ok=True)
    (out / "levels").mkdir(parents=True, exist_ok=True)
    iso = list(csv.DictReader(open(Path(a.out) / "isotopes.csv")))
    lev_cache = {}
    nc = nl = 0
    for r in iso:
        sym, A = r["symbol"], int(r["mass_number"])
        name = f"{sym}{A}"
        res = build_capture(a.egaf, name, out / "capture" / f"{name}.csv")
        if res and res["status"] == "ok":
            nc += 1
            print(f"  capture {name:6s} primary coverage {min(res['prim'],1):.3f}  E_gamma/Sn {res['E/Sn']:.3f}  {res['ntrans']} lines")
        else:
            print(f"  capture {name:6s} -> generic cascade ({'rejected' if res else 'not in EGAF'})")
            (out / "capture" / f"{name}.csv").unlink(missing_ok=True)
        lf = Path(a.levels) / f"{sym}.lev"
        if lf.exists():
            if sym not in lev_cache:
                lev_cache[sym] = parse_lev_file(lf)
            from_z = None
            for (z, mass), lv in lev_cache[sym].items():
                if mass == A and lv:
                    n = write_levels(lv, out / "levels" / f"{name}.csv")
                    nl += 1
                    print(f"  levels  {name:6s} {n} levels kept")
                    break
    print(f"capture cascades: {nc}, level schemes: {nl}")


if __name__ == "__main__":
    main()
