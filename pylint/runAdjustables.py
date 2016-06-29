#!/usr/bin/env python
from __future__ import print_function
from readyaml   import read_yaml
import subprocess

adjustables = read_yaml('adjustables.yml')['adjustables']

for adjustable in adjustables:
    setting = adjustables[adjustable]
    commands = ['pylint3', '--disable=all',
                           '--enable=%s' % adjustable,
                           '--%s'        % setting,
                           '--reports=n',
                           '../packages/pygsti']
    try:
        output = subprocess.check_output(commands)
        print(output.decode('utf-8'))
    except subprocess.CalledProcessError as e:
        print(e.output.decode('utf-8'))
