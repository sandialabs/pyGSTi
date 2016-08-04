#!/bin/bash
cd ../packages/pygsti/

for filename in `ls *`; do
    echo "============ Running $filename ==========================="
    clonedigger $filename -o "../../test/output/dup_$filename.html"
    echo ""
done
echo "DONE WITH TESTS"
