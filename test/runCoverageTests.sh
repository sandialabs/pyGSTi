#!/bin/bash
echo "Tests started..."
nosetests -v --with-coverage --cover-package=pygsti --cover-erase test*.py  > coverage_tests.out 2>&1
echo "Output written to coverage_tests.out"
