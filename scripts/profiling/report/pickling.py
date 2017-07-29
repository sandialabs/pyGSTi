#!/usr/bin/env python3
import pickle
import pygsti
import plotly
import json

from pygsti.tools import timed_block

def main():
    with open('data/full_report_results.pkl', 'rb') as infile:
        results_tp, results_full = pickle.load(infile)
    with timed_block('TP/Full multi report'):
        ws = pygsti.report.create_general_report({'TP': results_tp, "Full": results_full},
                                                "tutorial_files/exampleMultiGenReport.html",verbosity=3,
                                                auto_open=False)
    with open('data/testws.pkl', 'wb') as outfile:
        #pickle.dump(ws.smartCache, outfile)
        pickle.dump(ws, outfile)
    with open('data/testws.pkl', 'rb') as infile:
        ws = pickle.load(infile)
        #pickledCache = pickle.load(infile)
    #ws.smartCache = pickledCache
    with timed_block('reused ws'):
        pygsti.report.create_general_report({'TP': results_tp, "Full": results_full},
                                                "tutorial_files/exampleMultiGenReport.html",verbosity=3,
                                                auto_open=False, ws=ws)

if __name__ == '__main__':
    main()
