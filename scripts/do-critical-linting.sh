#!/bin/bash

SCRIPT_PATH=${0%/*}
pushd "$SCRIPT_PATH/.." > /dev/null
python3 -m flake8 --statistics --config=.flake8-critical pygsti
popd > /dev/null
