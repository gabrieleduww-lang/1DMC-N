#!/usr/bin/env python3
"""Build pointwise (continuous-energy) neutron data for 1DMC-N from ENDF/B-VIII.0.

Source: https://github.com/openmc-data-storage/ENDF-B-VIII.0-NNDC-json (294 K pointwise data, reconstructed
resonances, linearly interpolable). No energy-group collapse is done, so resonance self-shielding is handled
by the transport itself.

File format data/neutron/<iso>.nbin (little endian):
  char[8] "MCSHNP01"; u32 nE; u32 nCh; f64 A_ratio; f64 Sn_MeV
  f64 E[nE] (eV, <= 20 MeV)
  nCh x { i32 type; i32 ripl_level; f64 Ex_MeV; u32 start; u32 count; f32 xs[count] }   (xs for E[start:start+count], barn)
  type: 0 elastic, 1 capture, 2 other absorption (n,p),(n,a)..., 3 (n,2n), 4 continuum inelastic + other
        neutron-emitting residual, 5 discrete inelastic level (Ex = excitation energy; ripl_level = matching RIPL level or -1)

Usage: python3 tools/build_neutron.py --endf ENDF-B-VIII.0-NNDC-json-main.zip --out data
(run build_data.py --skip-neutron and build_gamma.py first: needs isotopes.csv and gamma/levels/*.csv)
"""
import argparse
import csv
import json
import re
import struct
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

EMAX = 2.0e7
TOL = 0.002           # grid thinning tolerance, relative to the local total cross section


def index_zip(z):
    pat = re.compile(rb'"Mass number": (\d+).*?"Atomic symbol": "(\w+)".*?"MT reaction number": (\d+)', re.S)
    idx = defaultdict(dict)
    for n in z.namelist():
        if not n.endswith(".json") or "index" in n:
            continue
        with z.open(n) as f:
            h = f.read(400)
        m = pat.search(h)
        if m:
            idx[(m.group(2).decode(), int(m.group(1)))][int(m.group(3))] = n
    return idx


def thin(E, Y, tolabs):
    """Greedy chord thinning: keep a subset of points so linear interpolation reproduces every column of Y
    within tolabs (per-point absolute tolerance)."""
    n = len(E)
    keep = [0]
    i = 0

    def ok(i, k):
        if k - i < 2:
            return True
        f = (E[i + 1:k] - E[i]) / (E[k] - E[i])
        chord = Y[i] + f[:, None] * (Y[k] - Y[i])
        return bool(np.all(np.abs(chord - Y[i + 1:k]) <= tolabs[i + 1:k, None]))

    while i < n - 1:
        good, step = i + 1, 1
        while True:                                    # gallop
            k = min(i + 2 * step, n - 1)
            if k == good:
                break
            if ok(i, k):
                good = k
                step *= 2
                if k == n - 1:
                    break
            else:
                lo, hi = good, k                        # binary search between good and k
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    if ok(i, mid):
                        lo = mid
                    else:
                        hi = mid
                good = lo
                break
        keep.append(good)
        i = good
    return np.array(keep)


def read_ripl_levels(path):
    if not Path(path).exists():
        return []
    out = []
    for line in open(path):
        if line.startswith("L,"):
            _, idx, e = line.strip().split(",")
            out.append((int(idx), float(e)))
    return out


def build(z, idx, sym, A, a_ratio, sn, ripl, out):
    files = idx[(sym, A)]

    def load(mt):
        d = json.loads(z.read(files[mt]))
        return np.array(d["energy"]), np.array(d["cross section"])

    E1, t1 = load(1)
    m = E1 <= EMAX
    E, tot = E1[m], t1[m]
    on = lambda mt: np.interp(E, *load(mt), left=0.0, right=0.0) if mt in files else np.zeros_like(E)
    el, cap = on(2), on(102)
    oabs = sum((on(mt) for mt in range(103, 118) if mt in files), np.zeros_like(E))
    n2n = on(16)
    lev_mts = [mt for mt in range(51, 91) if mt in files]
    levs = [on(mt) for mt in lev_mts]
    inel = on(4) if 4 in files else sum(levs, np.zeros_like(E)) + (on(91) if 91 in files else 0.0)
    cont = np.clip(inel - sum(levs, np.zeros_like(E)), 0, None)
    resid = np.clip(tot - (el + inel + n2n + cap + oabs), 0, None)
    resid[resid < 1e-3 * tot] = 0.0                       # drop rounding noise
    cont = cont + resid                                   # (n,np), (n,n alpha), (n,3n), ... treated as emitting a neutron

    chans = [(0, -1, 0.0, el), (1, -1, 0.0, cap), (2, -1, 0.0, oabs), (3, -1, 0.0, n2n), (4, -1, 0.0, cont)]
    for mt, y in zip(lev_mts, levs):
        if not np.any(y > 0):
            continue
        thr = E[np.argmax(y > 0)]
        ex = thr * 1e-6 * a_ratio / (a_ratio + 1)         # MeV
        best = None
        for li, le in ripl:
            if li > 0 and (best is None or abs(le - ex) < abs(best[1] - ex)):
                best = (li, le)
        match = best[0] if best and abs(best[1] - ex) <= max(0.008, 0.004 * ex) else -1
        chans.append((5, match, (best[1] if match >= 0 else ex), y))

    Y = np.column_stack([tot] + [c[3] for c in chans])
    keep = thin(E, Y, np.maximum(TOL * tot, 1e-7))
    Ek = E[keep]
    path = Path(out) / "neutron" / f"{sym}{A}.nbin"
    path.parent.mkdir(parents=True, exist_ok=True)
    wr = []
    for typ, ripl_idx, ex, y in chans:
        yk = y[keep].astype(np.float32)
        nz = np.nonzero(yk > 0)[0]
        if len(nz) == 0:
            continue
        start = max(0, int(nz[0]) - 1)
        wr.append((typ, ripl_idx, ex, start, yk[start:]))
    with open(path, "wb") as f:
        f.write(b"MCSHNP01")
        f.write(struct.pack("<IIdd", len(Ek), len(wr), a_ratio, sn))
        f.write(Ek.astype("<f8").tobytes())
        for typ, ri, ex, start, arr in wr:
            f.write(struct.pack("<iidII", typ, ri, ex, start, len(arr)))
            f.write(arr.astype("<f4").tobytes())
    return len(E), len(Ek), sum(1 for c in wr if c[0] == 5), sum(1 for c in wr if c[0] == 5 and c[1] >= 0), path.stat().st_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endf", required=True)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    z = zipfile.ZipFile(a.endf)
    idx = index_zip(z)
    tot_bytes = 0
    for r in csv.DictReader(open(Path(a.out) / "isotopes.csv")):
        sym, A = r["symbol"], int(r["mass_number"])
        if (sym, A) not in idx:
            print("  !! missing ENDF data for", sym, A)
            continue
        ripl = read_ripl_levels(Path(a.out) / "gamma" / "levels" / f"{sym}{A}.csv")
        n0, n1, nl, nm, sz = build(z, idx, sym, A, float(r["atomic_mass_amu"]) / 1.00866491588, float(r["Sn_MeV"]), ripl, a.out)
        tot_bytes += sz
        print(f"  {sym}{A:<4d} {n0:6d} -> {n1:6d} pts  {nl:2d} levels ({nm} matched to RIPL)  {sz / 1e3:7.0f} kB")
    print(f"total {tot_bytes / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
