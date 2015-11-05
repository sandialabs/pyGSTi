#!/bin/bash
echo "Tests started..."
nosetests -v --with-coverage --cover-package=GST --cover-erase *.py  > coverage_tests.out 2>&1
echo "Output written to coverage_tests.out"
