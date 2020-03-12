#!/usr/bin/env python3
from pygsti.report import Notebook

def main():
    nb = Notebook()
    nb.add_markdown('# Replica of tutorial 20, built using Python')
    nb.add_code_file('templates/setup.py')
    nb.add_code_file('templates/workspace.py')
    nb.add_markdown('After running GST, a `Workspace` object can be used to interpret the results:')
    nb.add_code('ws.GatesVsTargetTable(gs1, tgt)')
    nb.add_code('ws.SpamVsTargetTable(gs2, tgt)')
    nb.add_code('ws.ColorBoxPlot(("chi2","logl"), gss, ds1, gs1, box_labels=True)')
    nb.add_code('''ws.FitComparisonTable(gss.Ls, results1.gatestring_structs['iteration'],
                         results1.estimates['default'].gatesets['iteration estimates'], ds1)''')
    nb.add_code('ws.FitComparisonTable(["GS1","GS2","GS3"], [gss, gss, gss], [gs1,gs2,gs3], ds1, x_label="GateSet")')
    nb.add_code('''ws.ChoiTable(gs3, display=('matrix','barplot'))''')
    nb.add_code('''ws.GateMatrixPlot(gs1['Gx'],scale=1.5, box_labels=True)
ws.GateMatrixPlot(pygsti.tools.error_generator(gs1['Gx'], tgt['Gx']), scale=1.5)''')
    nb.add_code('ws.ErrgenTable(gs3,tgt)')
    nb.add_code('''ws.PolarEigenvaluePlot([np.linalg.eigvals(gs2['Gx'])],["purple"],scale=1.5)''')
    nb.add_code('''ws.GateEigenvalueTable(gs2, display=('evals','polar'))''')
    nb.launch_new('20Replica.ipynb')

if __name__ == '__main__':
    main()
