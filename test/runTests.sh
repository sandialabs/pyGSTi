#!/bin/bash

for filename in `ls test*.py`; do
    echo "============ Running $filename ==========================="
    python "$filename"
    echo ""
done
echo "DONE WITH TESTS"
