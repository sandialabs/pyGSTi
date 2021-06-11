#!/usr/bin/env python3
import pickle

import pygsti
from pygsti.tools import timed_block


def main():
    with open('data/full_report_results.pkl', 'rb') as infile:
        results_tp, results_full = pickle.load(infile)
    ws = pygsti.report.Workspace('data/wscache.pkl')
    with timed_block('reused ws'):
        pygsti.report.create_general_report({'TP': results_tp, "Full": results_full},
                                                "tutorial_files/exampleMultiGenReport.html",verbosity=3,
                                                auto_open=False, ws=ws)

if __name__ == '__main__':
    main()
