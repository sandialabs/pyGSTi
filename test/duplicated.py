#!/usr/bin/env python3
import os
import subprocess

from helpers.automation_tools import directory, get_packages

exclude = ['__pycache__']

print('Generating html reports of duplicated code')

with directory('../pygsti'):
    for package in get_packages(os.getcwd()):
        if package not in exclude:
            print('Finding duplicated code in %s' % package)
            subprocess.call(['clonedigger', package, '-o', '../../test/output/dup/dup_%s.html' % package])

print('Done')
