#!/usr/bin/env python3
from pygsti.report import Notebook

import time

def main():


    nb = Notebook()
    nb.add_markdown('# Pygsti report\n(Created on {})'.format(time.strftime("%B %d, %Y")))
    nb.add_code_file('setup.py')
    nb.add_code_file('workspace.py')
    nb.add_markdown('### Summary')
    nb.add_code('ws.FitComparisonBarPlot(Ls, gssAllL, switchBd.gsAllL, eff_ds, switchBd.objective, \'L\')')

    '''
        %(goSwitchboard1)s

        <figure id="progressBarPlot" class='tbl'>
          <figcaption><span class="captiontitle">Model Violation summary.</span> <span class="captiondetail">The aggregate log-likelihood as a function of GST iteration.</span></figcaption>
          %(progressBarPlot)s
        </figure>

        <figure id="bestEstimateColorScatterPlot" class='tbl'>
          <figcaption><span class="captiontitle">Per-sequence model violation.</span> <span class="captiondetail"> Each point displays the goodness of fit for a single gate sequence.</span> </figcaption>
          %(bestEstimateColorScatterPlot)s
        </figure>

        <figure id="bestGatesetVsTargetTable_sum" class='tbl'>
          <figcaption><span class="captiontitle">Comparison of GST estimated gates to target gates.</span> <span class="captiondetail"> This table presents, for each of the gates, three different measures of distance or discrepancy from the GST estimate to the ideal target operation.  See text for more detail.</span></figcaption>
          %(bestGatesetVsTargetTable_sum)s
        </figure>
    '''

    nb.add_markdown('### Goodness')
    nb.add_markdown('### Gauge Invariant')
    nb.add_markdown('### Gauge Variant')
    nb.add_markdown('### Data Comparison')
    nb.add_markdown('### Input')
    nb.add_markdown('### Meta')
    nb.launch_as('AdvancedReport.ipynb')

if __name__ == '__main__':
    main()
