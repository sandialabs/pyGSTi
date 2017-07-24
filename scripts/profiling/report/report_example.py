#!/usr/bin/env python3
import pickle
import pygsti

def main():
    with open('data/example_report_results.pkl', 'rb') as infile:
        results = pickle.load(infile)
    pygsti.report.create_general_report(results, "report/exampleGenReport.html",
                                        verbosity=0, auto_open=False)

if __name__ == '__main__':
    main()
