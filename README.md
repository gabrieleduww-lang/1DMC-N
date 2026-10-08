# 1DMC-N

A 1D (slab) coupled neutron-photon Monte Carlo code in C++17, with a Tk GUI for setting up
and plotting runs. I wrote it for teaching and quick scoping of shielding problems. The
engine is one source file, has no dependencies, and prints JSON.

It is not a design or licensing tool. The physics is checked against analytic limits, energy
conservation and a few measured quantities (see Checks), but it has not been benchmarked
against MCNP, OpenMC, Geant4 or experiments. Do that before you trust any number.

## Quick start

### Linux / macOS

```bash
make
make check
pip install -r requirements.txt
python3 gui/1dmc-n_gui.py
```

### Windows (g++)

Run this in PowerShell from the project folder. It installs WINLIB (which provides g++) and
Python with winget, then builds `build\1dmc-n.exe`.

```powershell
winget install -e --id BrechtSanders.WinLibs.POSIX.UCRT --accept-source-agreements
winget install -e --id Python.Python.3.12 --accept-source-agreements
$env:Path += ";" + (Get-ChildItem -Path "$env:LOCALAPPDATA\**\mingw64\bin" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName)
New-Item -ItemType Directory -Force build | Out-Null
g++ -O3 -std=c++17 -fopenmp -D_USE_MATH_DEFINES src\1dmc-n.cpp -o build\1dmc-n.exe
pip install -r requirements.txt
python gui\1dmc-n_gui.py
```

### Examples

```bash
build/1dmc-n --layer polyethylene:15 --layer borated_pe_5:5 --layer lead:3 --source cf252
build/1dmc-n --layer concrete:50 --source mono --energy 14.1 --angle cosine --dose ISO
build/1dmc-n --layer lead:5 --source cs137
build/1dmc-n --material mypoly:0.95:H=0.14,C=0.86 --tsl mypoly:H1=hpoly --layer mypoly:20
build/1dmc-n --help
```

The engine looks for `data/` next to the build directory. Use `--data DIR` or
`$ONEDMC_N_DATA` to point elsewhere. Speed is roughly 20-50k histories/s per core for
hydrogenous shields.

## What it models

- **Geometry and sources:** a stack of infinite slabs with the source on the left face.
  Sources are mono-energetic or thermal neutrons, Cf-252 or U-235 Watt spectra, mono-energetic
  photons, Cs-137 or Co-60, as a normal beam or with cosine-law incidence.
- **Neutrons:** continuous-energy ENDF/B-VIII.0 at 294 K for 66 isotopes (H to Pb), with
  resonances kept, so self-shielding comes out of the transport. Reactions are elastic,
  each discrete inelastic level, continuum inelastic, (n,2n), capture and other absorption.
- **Thermal scattering:** below about 5 eV, hydrogen in water uses the ENDF S(alpha,beta)
  table and hydrogen in polyethylene, paraffin and borated PE uses a table sampled from
  NCrystal. All other nuclides use free-gas motion below 400 kT.
- **Secondary photons:** capture gammas from measured EGAF cascades for 51 isotopes (generic
  energy-conserving cascade otherwise), and inelastic gammas from RIPL-3 level schemes.
- **Photons:** photoelectric (local deposit), Compton (Klein-Nishina) and pair production.
  No Rayleigh scattering.
- **Dose:** ICRP-116 effective dose per fluence (AP, PA, LLAT, RLAT, ROT or ISO), reported
  for neutrons, photons and the sum.
- **Variance reduction:** implicit capture with Russian roulette (the default) and optional
  geometric splitting with `--split R`.
- **Tallies:** transmission, reflection and absorption (mean and standard error), uncollided
  transmission against the analytic value, flux versus depth, spectra and dose ratios.
- **Comparison switches:** `--no-tsl`, `--generic-gamma`, `--no-gamma`, `--analog`.

## Checks

`make check` (or `python tests\check.py`) verifies:

1. Uncollided transmission matches exp(-Sigma t) for neutrons and photons.
2. Particle balance, including (n,2n) and pair production.
3. H(n,gamma) gives the 2.22 MeV line, and Fe-56 at 1.2 MeV emits the 846.8 keV line.
4. Implicit and analog capture agree, and split and unsplit runs agree.
5. Thermal neutron relaxation length in water is 2.88 cm against a measured 2.7-2.8 cm
   (3.7 cm with free-gas hydrogen).
6. The iron 24 keV window is resolved.
7. Capture cascades conserve energy (Fe-56, Pb-207, Si-28) to within 0.01 MeV.
8. Photon cross sections match NIST XCOM to within 3 %.

This shows the engine is self-consistent. It does not validate deep-penetration results,
secondary-gamma dose or polyethylene thermal behaviour against experiment.

## Limitations

- **Gamma data is uneven.** EGAF coverage is high for Fe, Si, Ca, Al, Na, Ni, Cu and
  Pb-207 (80-100 %) but low for Cd, Gd and W (roughly 10-25 %), where the generic cascade
  is too hard. Pb-208, B-11, O-18 and several Cd, Gd and W isotopes are fully generic. Each
  run reports `capture_measured_cascade_fraction`.
- **Cascades** are the thermal ones, reused for resonance and fast captures. Inelastic
  levels are matched to RIPL by energy, and continuum inelastic is a crude evaporation
  model with no angular correlations.
- **Thermal scattering** exists only for hydrogen in water and polyethylene. The
  polyethylene table is not an ENDF evaluation and its 2.1 cm relaxation length has not
  been checked against a reference. Everything else is free-gas, with no Bragg scattering.
- **Neutron data** is 294 K only, has no probability tables, and elastic scattering is
  isotropic in the CM frame. There is no fission.
- **Photons** have no bremsstrahlung, secondary electrons, photoneutrons or fluorescence,
  and photoelectric data above 0.5 MeV is an extrapolation (matters for high Z).
- **Geometry** is 1D, so no leakage, streaming or ducts. ICRP-116 coefficients are for the
  reference phantom, so a survey meter reading H*(10) will differ.

## Data

`data/` is generated by `tools/fetch_and_build.sh`. It downloads about 600 MB and needs git,
curl and `pip install numpy periodictable xraydb ncrystal`. Each builder script documents
its own file format.

## AI Disclosure & Attribution

Some portions of the C++ physics engine and Tk GUI logic were created with the assistance of artificial intelligence tools. All core transport algorithms, cross-section data parsers, and physics conservation laws have been independently verified, benchmarked against analytic limits, and subjected to automated regression testing (see the `Checks` section).

## Splitting for thick shields

`--split R` divides each layer into cells (default total/20, set with `--cell`) and gives
cell k an importance of R^k. Choose R so that R^(number of cells) is about the expected
attenuation, and compare with an unsplit run on a thinner shield first.

## License

MIT for the code, see `LICENSE`.
