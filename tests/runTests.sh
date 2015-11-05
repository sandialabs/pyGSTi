#!/bin/bash

for filename in `ls *Tests.py`; do
    echo "============ Running $filename ==========================="
    python "$filename"
    echo ""
done
echo "DONE WITH TESTS"
