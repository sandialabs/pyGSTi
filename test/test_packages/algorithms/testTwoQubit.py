import unittest
import os, sys

from ..testutils import BaseTestCase, compare_files, temp_files

class TestTwoQubitMethods(BaseTestCase):

    def setUp(self):
        super(TwoQubitTestCase, self).setUp()

        #Set Model objects to "non-strict" mode for this testing
        pygsti.objects.ExplicitOpModel._strict = False


    def runTwoQubit(self):
        #The two-qubit model
        target_model = pygsti.construction.build_explicit_model(
            [4], [('Q0','Q1')],['Gix','Giy','Gxi','Gyi','Gcnot'],
            [ "X(pi/2,Q1)", "Y(pi/2,Q1)", "X(pi/2,Q0)", "Y(pi/2,Q0)", "CX(pi,Q0,Q1)" ],
            ["rho0"], ["0"], ["E0","E1","E2","Ec"], ["0","1","2","C"],
            spamdefs={'upup': ("rho0","E0"), 'updn': ("rho0","E1"),
                      'dnup': ("rho0","E2"), 'dndn': ("rho0","Ec") },
            basis="gm" )

        fiducialStrings16 = pygsti.construction.circuit_list(
            [ (), ('Gix',), ('Giy',), ('Gix','Gix'),
              ('Gxi',), ('Gxi','Gix'), ('Gxi','Giy'), ('Gxi','Gix','Gix'),
              ('Gyi',), ('Gyi','Gix'), ('Gyi','Giy'), ('Gyi','Gix','Gix'),
              ('Gxi','Gxi'), ('Gxi','Gxi','Gix'), ('Gxi','Gxi','Giy'), ('Gxi','Gxi','Gix','Gix') ] )

        specs16 = pygsti.construction.build_spam_specs(
            fiducialStrings16, prep_labels=['rho0'],
            effect_labels=['E0','E1','E2','Ec'])

        germs4 = pygsti.construction.circuit_list(
            [ ('Gix',), ('Giy',), ('Gxi',), ('Gyi',) ] )

        #Run min-chi2 GST
        # To run for longer, add powers of 2 to maxLs (e.g. [1,2,4], [1,2,4,8], etc)
        gsets1, dsGen1 = self.runMC2GSTAnalysis(
            specs16, germs4, target_model, 1234, maxLs = [1,2], nSamples=1000)

if __name__ == "__main__":
    unittest.main(verbosity=2)
