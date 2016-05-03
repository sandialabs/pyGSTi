import unittest

class TwoQubitTestCase(unittest.TestCase):

    def setUp(self):
        #Set GateSet objects to "non-strict" mode for this testing
        pygsti.objects.GateSet._strict = False

    def runSilent(self, callable, *args, **kwds):
        orig_stdout = sys.stdout
        sys.stdout = open("temp_test_files/silent.txt","w")
        result = callable(*args, **kwds)
        sys.stdout.close()
        sys.stdout = orig_stdout
        return result


class TestTwoQubitMethods(TwoQubitTestCase):
    
    
    def runTwoQubit(self):
        #The two-qubit gateset
        gs_target = pygsti.construction.build_gateset( 
            [4], [('Q0','Q1')],['Gix','Giy','Gxi','Gyi','Gcnot'], 
            [ "X(pi/2,Q1)", "Y(pi/2,Q1)", "X(pi/2,Q0)", "Y(pi/2,Q0)", "CX(pi,Q0,Q1)" ],
            ["rho0"], ["0"], ["E0","E1","E2"], ["0","1","2"], 
            spamdefs={'upup': ("rho0","E0"), 'updn': ("rho0","E1"),
                      'dnup': ("rho0","E2"), 'dndn': ("rho0","remainder") },
            basis="gm" )
        
        fiducialStrings16 = pygsti.construction.gatestring_list( 
            [ (), ('Gix',), ('Giy',), ('Gix','Gix'), 
              ('Gxi',), ('Gxi','Gix'), ('Gxi','Giy'), ('Gxi','Gix','Gix'), 
              ('Gyi',), ('Gyi','Gix'), ('Gyi','Giy'), ('Gyi','Gix','Gix'), 
              ('Gxi','Gxi'), ('Gxi','Gxi','Gix'), ('Gxi','Gxi','Giy'), ('Gxi','Gxi','Gix','Gix') ] )
        
        specs16 = pygsti.construction.build_spam_specs(
            fiducialStrings16, prep_labels=['rho0'], 
            effect_labels=['E0','E1','E2', 'remainder'])
        
        germs4 = pygsti.construction.gatestring_list(
            [ ('Gix',), ('Giy',), ('Gxi',), ('Gyi',) ] )
    
        #Run min-chi2 GST
        # To run for longer, add powers of 2 to maxLs (e.g. [1,2,4], [1,2,4,8], etc)
        gsets1, dsGen1 = self.runMC2GSTAnalysis(
            specs16, germs4, gs_target, 1234, maxLs = [1,2], nSamples=1000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
