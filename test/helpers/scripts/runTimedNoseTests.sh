#!/bin/bash
# nosetests -v */*.py mpi/mpitest*.py 2>&1 | tee out.txt

cd ../..

for filename in `ls */test*.py`; do
    echo "============ Running $filename ==========================="
    time nosetests -v $filename
    echo ""
done
echo "DONE WITH TESTS"
