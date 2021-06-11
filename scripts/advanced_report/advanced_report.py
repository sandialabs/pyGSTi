#!/usr/bin/env python3
import time

from pygsti.report import Notebook


def main():
    nb = Notebook()
    nb.add_markdown('# Pygsti report\n(Created on {})'.format(time.strftime("%B %d, %Y")))
    nb.add_code_file('templates/setup.py')
    nb.add_code_file('templates/workspace.py')
    nb.add_notebook_text_files([
        'templates/summary.txt',
        'templates/goodness.txt',
        'templates/gauge_invariant.txt',
        'templates/gauge_variant.txt',
        'templates/data_comparison.txt',
        'templates/input.txt',
        'templates/meta.txt'])
    nb.launch_new('AdvancedReport.ipynb')

if __name__ == '__main__':
    main()
