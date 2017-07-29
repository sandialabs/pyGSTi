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
        pickleable = dict()
        for k, v in ws.smartCache.cache.items():
            try:
                pickle.dumps(v, protocol=2)
                pickleable[k] = v
            except TypeError as e:
                pass
                '''
                print('{:<45}({}) failed to pickle: {}'.format(k[0], type(v), e))
                #print(v)
                print(dir(v))
                try:
                    rep = str(v)
                    print(rep)
                    rep = rep.replace('\'', '"')
                    print(rep)
                    figJSON = json.loads(rep)
                    print(figJSON)
                    print(plotly.offline.plot(figJSON))
                except Exception as e:
                    print(e)
                '''
        pickle.dump(pickleable, outfile)

        print(list(pickleable.keys()))

    '''
    with timed_block('reused ws'):
        pygsti.report.create_general_report({'TP': results_tp, "Full": results_full},
                                                "tutorial_files/exampleMultiGenReport.html",verbosity=3,
                                                auto_open=False, ws=ws)
    '''

if __name__ == '__main__':
    main()
