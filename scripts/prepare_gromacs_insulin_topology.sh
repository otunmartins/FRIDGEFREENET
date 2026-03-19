#!/usr/bin/env bash
# One-off: regenerate committed protein topology for 4F1C (optional).
# Default pipeline runs pdb2gmx at evaluate time; use this to debug SS bonds.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PDB="$REPO/src/python/insulin_ai/simulation/data/4F1C.pdb"
OUT="$REPO/src/python/insulin_ai/simulation/data/gromacs"
mkdir -p "$OUT"
cd "$OUT"
GMX="${GMX_BIN:-gmx}"
"$GMX" pdb2gmx -f "$PDB" -o protein.gro -p protein.top -i posre.itp \
  -water tip3p -ff amber99sb-ildn -ignh <<'EOF'
y
y
y
y
y
y
y
y
y
y
y
y
y
y
y
EOF
echo "Wrote $OUT/protein.{gro,top}"
