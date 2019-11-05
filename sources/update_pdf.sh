#! /usr/bin/env bash

# Don't forget to echo "CACHE_DIRECTORY = '$(pwd)/assets'" >> ~/.grip/settings.py

export GRIPURL=$(pwd)

for f in *.md; do grip --export $f; done
for f in *.html; do wkhtmltopdf $f ${f%.*}.pdf; done
rm -f *.html

mv ./Subject-Part0-*.pdf ../0-deviceQuery/
mv ./Subject-Part1-*.pdf ../1-Obfuscation/
mv ./Subject-Part2-*.pdf ../2-Convolutions/
