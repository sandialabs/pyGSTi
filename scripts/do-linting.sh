#!/bin/bash

SCRIPT_PATH=${0%/*}
pushd "$SCRIPT_PATH/.." > /dev/null
python3 -m flake8 --statistics packages/pygsti
popd > /dev/null
