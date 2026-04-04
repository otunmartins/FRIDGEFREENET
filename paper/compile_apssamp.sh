#!/usr/bin/env bash
# Build paper/apssamp.tex (fairmeta). Run from repository root or this directory.
set -euo pipefail
cd "$(dirname "$0")"
export TEXINPUTS="$(pwd)/../docs:${TEXINPUTS:-.//:}"
pdflatex -interaction=nonstopmode apssamp.tex
bibtex apssamp
pdflatex -interaction=nonstopmode apssamp.tex
pdflatex -interaction=nonstopmode apssamp.tex
