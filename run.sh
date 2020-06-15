#!/usr/bin/env bash
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/code

echo $PYTHONPATH

python3 code/main.py
