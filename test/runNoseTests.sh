#!/bin/bash
nosetests -v test*.py mpitest*.py 2>&1 | tee out.txt
