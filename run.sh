#!/usr/bin/env bash
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/code

echo $PYTHONPATH

python3 code/recall_main.py

python3 code/ranking_main.py
