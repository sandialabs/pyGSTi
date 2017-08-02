import unittest
import pygsti
import sys, os

import pygsti.algorithms.germselection as germsel
import pygsti.algorithms.fiducialselection as fidsel
import pygsti.construction as constr


from pygsti.construction import std1Q_XYI as std

from ..testutils import BaseTestCase, compare_files, temp_files

class AutoExperimentDesignTestCase(BaseTestCase):

    def setUp(self):
        super(AutoExperimentDesignTestCase, self).setUp()

    def test_auto_experiment_design(self):
        # Let's construct a 1-qubit $X(\pi/2)$, $Y(\pi/2)$, $I$ gateset for which we will need to find germs and fiducials.

        gs_target = constr.build_gateset([2], [('Q0',)], ['Gi', 'Gx', 'Gy'],
                                         ["I(Q0)", "X(pi/2,Q0)", "Y(pi/2,Q0)"],
                                         prepLabels=['rho0'], prepExpressions=["0"],
                                         effectLabels=['E0'], effectExpressions=["1"],
                                         spamdefs={'plus': ('rho0', 'E0'),
                                                   'minus': ('rho0', 'remainder')})


        # ## Hands-off

        # We begin by demonstrating the most hands-off approach.

        # We can generate a germ set simply by providing the target gateset.


        germs = germsel.generate_germs(gs_target)


        # In the same way we can generate preparation and measurement fiducials.


        prepFiducials, measFiducials = fidsel.generate_fiducials(gs_target)

        #test returnAll - this just prints more info...
        p,m = fidsel.generate_fiducials(gs_target, algorithm_kwargs={'returnAll': True})


        # Now that we have germs and fiducials, we can construct the list of experiments we need to perform in
        # order to do GST. The only new things to provide at this point are the sizes for the experiments we want
        # to perform (in this case we want to perform between 0 and 256 gates between fiducial pairs, going up
        # by a factor of 2 at each stage).


        maxLengths = [0] + [2**n for n in range(8 + 1)]
        listOfExperiments = constr.make_lsgst_experiment_list(gs_target.gates.keys(), prepFiducials,
                                                              measFiducials, germs, maxLengths)


        # The list of `GateString` that the previous function gave us isn't necessarily the most readable
        # form to present the information in, so we can write the experiment list out to an empty data
        # file to be filled in after the experiments are performed.

        graspGerms = germsel.generate_germs(gs_target, algorithm='grasp', algorithm_kwargs={'iterations': 1})
        slackPrepFids, slackMeasFids = fidsel.generate_fiducials(gs_target, algorithm='slack',
                                                                 algorithm_kwargs={'slackFrac': 0.25})


        max([len(germ) for germ in germs])

        germsMaxLength5 = germsel.generate_germs(gs_target, maxGermLength=5)

        max([len(germ) for germ in germsMaxLength5])

        germsMaxLength3 = germsel.generate_germs(gs_target, maxGermLength=3)

        uniformPrepFids, uniformMeasFids = fidsel.generate_fiducials(gs_target, maxFidLength=3,
                                                                     algorithm='grasp',
                                                                     algorithm_kwargs={'iterations': 100})


        incompletePrepFids, incompleteMeasFids = fidsel.generate_fiducials(gs_target, maxFidLength=1)

        nonSingletonGerms = germsel.generate_germs(gs_target, force=None, maxGermLength=4,
                                                   algorithm='grasp', algorithm_kwargs={'iterations': 5})


        omitIdentityPrepFids, omitIdentityMeasFids = fidsel.generate_fiducials(gs_target, omitIdentity=False,
                                                                               gatesToOmit=['Gi'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
