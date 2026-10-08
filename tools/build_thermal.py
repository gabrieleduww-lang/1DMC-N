#!/usr/bin/env python3
"""Build bound-atom thermal-scattering tables (S(alpha,beta)-derived) for 1DMC-N.

Table format (data/thermal/<name>.tsl, little endian):
  char[8] "MCSHTSL1"; u32 nE; u32 nEout; u32 nMu; f64 Emax_eV; f64 awr
  f64 E[nE] (eV); f32 sigma[nE] (barn per bound atom); f32 Eout[nE][nEout] (eV); f32 mu[nE][nEout][nMu]
Sampling rule (identical to ENDF/ACE "skewed" tables): pick outgoing-energy bin j with relative
weights 0.1, 0.4, 1, ..., 1, 0.4, 0.1; pick one of the nMu equiprobable cosines; interpolate E' and mu
between incident-energy rows i and i+1.

Modes
  ace       parse a thermal ACE file produced by NJOY (e.g. marquezj/tsl-HinH2O, ENDF/B-VIII.0 CAB model)
  ncrystal  sample NCrystal scattering kernels (e.g. Polyethylene_CH2.ncmat) and bin the samples

Examples
  python3 tools/build_thermal.py ace --file lib_294.ace --name hh2o --out data/thermal
  python3 tools/build_thermal.py ncrystal --cfg "Polyethylene_CH2.ncmat;temp=293.6K" --natoms 3 --other-b 4.75 \
          --name hpoly --out data/thermal
"""
import argparse
import struct
from pathlib import Path

import numpy as np

NEOUT, NMU = 200, 32
W = np.ones(NEOUT)
W[0] = W[-1] = 0.1
W[1] = W[-2] = 0.4
W = W / W.sum()                       # skewed outgoing-energy bin probabilities


def write_tsl(path, E, sigma, Eout, mu, emax, awr):
    nE, nEo, nMu = Eout.shape[0], Eout.shape[1], mu.shape[2]
    with open(path, "wb") as f:
        f.write(b"MCSHTSL1")
        f.write(struct.pack("<IIIdd", nE, nEo, nMu, emax, awr))
        f.write(np.asarray(E, "<f8").tobytes())
        f.write(np.asarray(sigma, "<f4").tobytes())
        f.write(np.asarray(Eout, "<f4").tobytes())
        f.write(np.asarray(mu, "<f4").tobytes())


# ----------------------------------------------------------------------------------- ACE
def read_ace(path):
    lines = Path(path).read_text().splitlines()
    awr = float(lines[0].split()[1])
    nxs = [int(x) for x in " ".join(lines[6:8]).split()]
    jxs = [int(x) for x in " ".join(lines[8:12]).split()]
    xss = np.array(" ".join(lines[12:]).split(), dtype=float)
    return awr, nxs, jxs, xss


def from_ace(path):
    awr, nxs, jxs, xss = read_ace(path)
    nil, nieb, ifeng = nxs[2], nxs[3], nxs[6]
    assert nxs[1] == 3 and ifeng == 1 and nieb == NEOUT, f"unexpected ACE layout NXS={nxs[:8]}"
    nmu = nil + 1
    i0 = jxs[0] - 1
    nE = int(xss[i0])
    E = xss[i0 + 1:i0 + 1 + nE] * 1e6                      # MeV -> eV
    sig = xss[i0 + 1 + nE:i0 + 1 + 2 * nE]
    assert jxs[2] - 1 == i0 + 1 + 2 * nE, "inelastic distribution does not follow cross sections"
    blk = xss[jxs[2] - 1:jxs[2] - 1 + nE * nieb * (nmu + 1)].reshape(nE, nieb, nmu + 1)
    Eout, mu = blk[:, :, 0] * 1e6, blk[:, :, 1:]
    assert jxs[3] == 0, "elastic scattering present; not supported here"
    return E, sig, Eout, mu, float(E[-1]), awr


# ------------------------------------------------------------------------------ NCrystal
def bin_samples(Ep, mu):
    order = np.argsort(Ep)
    Ep, mu = Ep[order], mu[order]
    n = len(Ep)
    cum = np.concatenate([[0], np.cumsum(W)])
    edges = np.round(cum * n).astype(int)
    eo = np.empty(NEOUT)
    mo = np.empty((NEOUT, NMU))
    for j in range(NEOUT):
        a, b = edges[j], max(edges[j + 1], edges[j] + 1)
        eo[j] = Ep[a:b].mean()
        m = np.sort(mu[a:b])
        k = np.round(np.linspace(0, len(m), NMU + 1)).astype(int)
        for c in range(NMU):
            seg = m[k[c]:max(k[c + 1], k[c] + 1)]
            mo[j, c] = seg.mean() if len(seg) else m[min(k[c], len(m) - 1)]
    return eo, mo


def from_ncrystal(cfg, natoms, other_b, emax=5.0, nE=96, nsamp=600000):
    import NCrystal as NC
    sc = NC.createScatter(cfg)
    E = np.logspace(-5, np.log10(emax), nE)
    sig = np.empty(nE)
    Eout = np.empty((nE, NEOUT))
    mu = np.empty((nE, NEOUT, NMU))
    nh = natoms - 1
    for i, e in enumerate(E):
        xs_total = sc.crossSectionIsotropic(e) * natoms        # barn per formula unit
        sig[i] = (xs_total - other_b) / nh                      # attributed to the bound H atoms
        ep, m = sc.sampleScatterIsotropic(e, repeat=nsamp)
        Eout[i], mu[i] = bin_samples(np.asarray(ep), np.asarray(m))
    return E, sig, Eout, mu, float(emax), 0.99917


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("ace"); a.add_argument("--file", required=True)
    n = sub.add_parser("ncrystal"); n.add_argument("--cfg", required=True)
    n.add_argument("--natoms", type=int, required=True, help="atoms per formula unit (3 for H2O and CH2)")
    n.add_argument("--other-b", type=float, required=True, help="free-atom scattering of the non-H atom(s), barn")
    for p in (a, n):
        p.add_argument("--name", required=True); p.add_argument("--out", default="data/thermal")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    if args.mode == "ace":
        E, sig, Eout, mu, emax, awr = from_ace(args.file)
    else:
        E, sig, Eout, mu, emax, awr = from_ncrystal(args.cfg, args.natoms, args.other_b)
    write_tsl(Path(args.out) / f"{args.name}.tsl", E, sig, Eout, mu, emax, awr)
    i = int(np.argmin(abs(E - 0.0253)))
    print(f"{args.name}: nE={len(E)} E=[{E[0]:.3g},{E[-1]:.3g}] eV  sigma({E[i]:.4f} eV)={sig[i]:.3f} b per H")


if __name__ == "__main__":
    main()
