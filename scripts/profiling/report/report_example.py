#!/usr/bin/env python3
import pickle

import pygsti
from pygsti.tools import timed_block


def main():
    with open('data/example_report_results.pkl', 'rb') as infile:
        results = pickle.load(infile)
    with timed_block('example report creation'):
        pygsti.report.create_general_report(results, "report/exampleGenReport.html",
                                            verbosity=0, auto_open=False)

if __name__ == '__main__':
    main()
