#!/usr/bin/bash

PYTHON_EXE=/NIRAL/work/munsell/toolkits/anaconda/anaconda3/bin/python ; export PYTHON_EXE

cmd="$PYTHON_EXE Enigma_pipeline.py -sdx $1 -gpu $2 -csv $3"

eval $cmd
