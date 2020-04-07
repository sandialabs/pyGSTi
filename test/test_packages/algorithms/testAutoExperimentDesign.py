import unittest
import pygsti
import sys, os

import pygsti.algorithms.germselection as germsel
import pygsti.algorithms.fiducialselection as fidsel
import pygsti.construction as constr


from pygsti.modelpacks.legacy import std1Q_XYI as std

from ..testutils import BaseTestCase, compare_files, temp_files

class AutoExperimentDesignTestCase(BaseTestCase):

    def setUp(self):
        super(AutoExperimentDesignTestCase, self).setUp()

    def test_auto_experiment_design(self):
        # Let's construct a 1-qubit $X(\pi/2)$, $Y(\pi/2)$, $I$ model for which we will need to find germs and fiducials.

        target_model = constr.build_explicit_model([('Q0',)], ['Gi', 'Gx', 'Gy'],
                                         ["I(Q0)", "X(pi/2,Q0)", "Y(pi/2,Q0)"])


        # ## Hands-off

        # We begin by demonstrating the most hands-off approach.

        # We can generate a germ set simply by providing the target model. (and seed so it's deterministic)
        
        germs = germsel.generate_germs(target_model, seed=2017)


        # In the same way we can generate preparation and measurement fiducials.


        prepFiducials, measFiducials = fidsel.generate_fiducials(target_model)

        #test returnAll - this just prints more info...
        p,m = fidsel.generate_fiducials(target_model, algorithm_kwargs={'returnAll': True})

        #test invalid algorithm
        with self.assertRaises(ValueError):
            fidsel.generate_fiducials(target_model, algorithm='foobar') 


        # Now that we have germs and fiducials, we can construct the list of experiments we need to perform in
        # order to do GST. The only new things to provide at this point are the sizes for the experiments we want
        # to perform (in this case we want to perform between 0 and 256 gates between fiducial pairs, going up
        # by a factor of 2 at each stage).


        maxLengths = [0] + [2**n for n in range(8 + 1)]
        listOfExperiments = constr.make_lsgst_experiment_list(target_model.operations.keys(), prepFiducials,
                                                              measFiducials, germs, maxLengths)


        # The list of `Circuit` that the previous function gave us isn't necessarily the most readable
        # form to present the information in, so we can write the experiment list out to an empty data
        # file to be filled in after the experiments are performed.

        graspGerms = germsel.generate_germs(target_model, algorithm='grasp', 
                                            seed=2017, numGSCopies=2,
                                            candidateGermCounts={3: 'all upto', 4:10, 5:10, 6:10},
                                            candidateSeed=2017,
                                            algorithm_kwargs={'iterations': 1})
        slackPrepFids, slackMeasFids = fidsel.generate_fiducials(target_model, algorithm='slack',
                                                                 algorithm_kwargs={'slackFrac': 0.25})
        fidsel.generate_fiducials(target_model, algorithm='slack') # slacFrac == 1.0 if don't specify either slackFrac or fixedSlack
                

        germsMaxLength3 = germsel.generate_germs(target_model, candidateGermCounts={3: 'all upto'}, seed=2017)

        uniformPrepFids, uniformMeasFids = fidsel.generate_fiducials(target_model, maxFidLength=3,
                                                                     algorithm='grasp',
                                                                     algorithm_kwargs={'iterations': 100})


        incompletePrepFids, incompleteMeasFids = fidsel.generate_fiducials(target_model, maxFidLength=1)

        nonSingletonGerms = germsel.generate_germs(target_model, numGSCopies=2, force=None, candidateGermCounts={4: 'all upto'},
                                                   algorithm='grasp', algorithm_kwargs={'iterations': 5},
                                                   seed=2017)


        omitIdentityPrepFids, omitIdentityMeasFids = fidsel.generate_fiducials(target_model, omitIdentity=False,
                                                                               opsToOmit=['Gi'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
