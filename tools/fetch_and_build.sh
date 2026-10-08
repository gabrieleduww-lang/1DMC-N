#!/usr/bin/env bash
# Rebuild everything in data/ from the original public sources.
# Needs: git, curl, python3 with numpy periodictable xraydb ncrystal (pip install ...).
# Downloads ~600 MB (ENDF JSON zip) and a sparse checkout of TALYS (levels only).
set -euo pipefail
WORK=${WORK:-/tmp/1dmc-n_src}; OUT=${OUT:-data}; HERE=$(cd "$(dirname "$0")/.." && pwd); mkdir -p "$WORK"; cd "$WORK"

# 1. ENDF/B-VIII.0 pointwise neutron data (JSON conversion of the NNDC library)
[ -f endf_json.zip ] || curl -L -o endf_json.zip \
  https://codeload.github.com/openmc-data-storage/ENDF-B-VIII.0-NNDC-json/zip/refs/heads/main
# 2. EGAF thermal-capture level schemes (JSON copy of the IAEA Evaluated Gamma-ray Activation File)
[ -d python_egaf ] || git clone --depth 1 https://github.com/AaronMHurst/python_egaf
# 3. RIPL-3 discrete levels as shipped with TALYS (MIT licence); only the levels/exp folder is fetched
if [ ! -d talys ]; then
  git clone --depth 1 --filter=blob:none --no-checkout https://github.com/arjankoning1/talys
  git -C talys sparse-checkout init --cone && git -C talys sparse-checkout set structure/levels/exp && git -C talys checkout
fi
# 4. ICRP-116 dose coefficients (as bundled with OpenMC) and the ENDF/B-VIII.0 H-in-H2O thermal ACE file
mkdir -p icrp116
for f in neutrons photons; do
  [ -f icrp116/$f.txt ] || curl -sL -o icrp116/$f.txt https://raw.githubusercontent.com/openmc-dev/openmc/develop/openmc/data/dose/icrp116/$f.txt
done
[ -d tsl-HinH2O ] || { git clone --depth 1 --filter=blob:none --no-checkout https://github.com/marquezj/tsl-HinH2O
  git -C tsl-HinH2O sparse-checkout init --no-cone && git -C tsl-HinH2O sparse-checkout set ace/lib_294.ace && git -C tsl-HinH2O checkout; }

cd "$HERE"
python3 tools/build_data.py    --icrp-dir "$WORK/icrp116" --out "$OUT"
python3 tools/build_gamma.py   --egaf "$WORK/python_egaf/pyEGAF/EGAF_JSON" --levels "$WORK/talys/structure/levels/exp" --out "$OUT"
python3 tools/build_neutron.py --endf "$WORK/endf_json.zip" --out "$OUT"
python3 tools/build_thermal.py ace --file "$WORK/tsl-HinH2O/ace/lib_294.ace" --name hh2o --out "$OUT/thermal"
python3 tools/build_thermal.py ncrystal --cfg "Polyethylene_CH2.ncmat;temp=293.6K" --natoms 3 --other-b 4.75 --name hpoly --out "$OUT/thermal"
echo "data built in $OUT"
