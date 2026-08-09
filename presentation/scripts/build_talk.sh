#!/bin/sh
# Builds both variants of the IAIFI talk from one source.
#
#   talk_iaifi_2026_v2_animated.pdf  autoplaying manifold animation.
#                                    Needs Acrobat (or pdfpc/Impressive).
#   talk_iaifi_2026_v2_static.pdf    same frames as click-through stills.
#                                    Works in Preview, Skim, browsers, Zoom.
#
# Present from _animated only on a machine you control that has Acrobat.
# Always carry _static as the fallback.
set -e
cd "$(dirname "$0")/.."

echo "==> animated"
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -jobname=talk_iaifi_2026_v2_animated \
  -pdflatex='pdflatex %O "\def\useanimations{1}\input{%S}"' \
  talk_iaifi_2026_v2.tex

echo "==> static"
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -jobname=talk_iaifi_2026_v2_static \
  talk_iaifi_2026_v2.tex

# Drop the aux/log/fls clutter, keep the two PDFs.
# -c leaves beamer's .nav/.snm/.toc behind, so remove those by hand.
echo "==> cleaning intermediates"
for job in talk_iaifi_2026_v2_animated talk_iaifi_2026_v2_static; do
  latexmk -c -jobname="$job" talk_iaifi_2026_v2.tex >/dev/null
  rm -f "$job".nav "$job".snm "$job".toc "$job".out
done

echo
ls -lh talk_iaifi_2026_v2_animated.pdf talk_iaifi_2026_v2_static.pdf
