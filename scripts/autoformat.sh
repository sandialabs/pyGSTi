#! /usr/bin/env bash
# Run code formatting tools across the codebase.
#
# About:
# This script makes two passes with autopep8.
# The first pass makes "nonaggressive" changes -- formatting changes
# which couldn't possibly introduce semantic differences.
# The second pass makes aggressive corrections for the following
# formatting errors:
#
# - W291: trailing whitespace
# - W292: no newline at end of file
# - W293: blank line contains whitespace
#
# In principle, these can potentially alter semantics. In practice,
# though, if this could possibly change a unit's semantics, that unit
# should probably be revised anyway. Still, it's not a bad idea to
# manually check over autoformatted files before committing.

PYGSTI=$(realpath $(dirname $(realpath $0))/..)

AUTOPEP8_BIN=$(which autopep8)
AUTOPEP8_BASIC_ARGS="--verbose --recursive --in-place"
AUTOPEP8_EXTRA_ARGS="--aggressive --select=W2"

if [ $? -ne 0 ]; then
    echo "No `autopep8` in path. Try running `pip install autopep8`."
    exit 1
fi

if [ $# -eq 0 ]; then
    TARGET="$PYGSTI"
else
    TARGET="$@"
fi

# General whitespace autoformatting
FORMAT_CMD="$AUTOPEP8_BIN $AUTOPEP8_BASIC_ARGS $TARGET"
echo "\$ $FORMAT_CMD"
$FORMAT_CMD

# Remove trailing whitespace, which requires --aggressive flag
FORMAT_CMD="$AUTOPEP8_BIN $AUTOPEP8_BASIC_ARGS $AUTOPEP8_EXTRA_ARGS $TARGET"
echo "\$ $FORMAT_CMD"
$FORMAT_CMD
