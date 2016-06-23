for filename in `ls */test*.py`; do
    echo "============ Running $filename ==========================="
    python3 $filename 2>&1
    echo ""
done
echo "DONE WITH TESTS"

