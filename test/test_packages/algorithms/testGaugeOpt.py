import unittest
import pygsti
from pygsti.construction import std1Q_XYI as std
from pygsti.baseobjs.basis import Basis

import numpy as np
from scipy import polyfit
import sys, os

from ..testutils import BaseTestCase, compare_files, temp_files
from ..algorithms.basecase import AlgorithmsBase

class TestCoreMethods(AlgorithmsBase):
    def test_gaugeopt_and_contract(self):
        ds = self.ds_lgst

        #pygsti.construction.generate_fake_data(self.datagen_gateset, self.lgstStrings,
        #                                            nSamples=10000,sampleError='binomial', seed=100)

        gs_lgst = pygsti.do_lgst(ds, self.fiducials, self.fiducials, self.gateset, svdTruncateTo=4, verbosity=0)


        #Gauge Opt to Target
        gs_lgst_target     = self.runSilent(pygsti.gaugeopt_to_target, gs_lgst, self.gateset, verbosity=10, checkJac=True)

        #
        gs_lgst.basis = self.gateset.basis.copy()
        gs_clgst_cp    = self.runSilent(pygsti.contract, gs_lgst, "CP",verbosity=10, tol=10.0, useDirectCP=False) #non-direct CP contraction

        #Gauge Opt to Target using non-frobenius metrics
        gs_lgst_targetAlt  = self.runSilent(pygsti.gaugeopt_to_target, gs_lgst_target, self.gateset,
                                            gatesMetric='fidelity', verbosity=10, checkJac=True)

        gs_lgst_targetAlt  = self.runSilent(pygsti.gaugeopt_to_target, gs_lgst_target, self.gateset,
                                            gatesMetric='tracedist', verbosity=10, checkJac=True)

        gs_lgst_targetAlt  = self.runSilent(pygsti.gaugeopt_to_target, gs_lgst_target, self.gateset,
                                            spamMetric='fidelity', verbosity=10, checkJac=True)

        gs_lgst_targetAlt  = self.runSilent(pygsti.gaugeopt_to_target, gs_lgst_target, self.gateset,
                                            spamMetric='tracedist', verbosity=10, checkJac=True)

        #Using other methods
        gs_BFGS = self.runSilent(pygsti.gaugeopt_to_target, gs_lgst, self.gateset, method='BFGS', verbosity=10)
        with self.assertRaises(ValueError): #Invalid metric
            self.runSilent(pygsti.gaugeopt_to_target, gs_lgst, self.gateset, method='BFGS', spamMetric='foobar', verbosity=10)
        with self.assertRaises(ValueError): #Invalid metric
            self.runSilent(pygsti.gaugeopt_to_target, gs_lgst, self.gateset, method='BFGS', gatesMetric='foobar', verbosity=10)
            
        with self.assertRaises(ValueError): #can't use least-squares for anything but frobenius metric
            self.runSilent(pygsti.gaugeopt_to_target, gs_lgst_target, self.gateset,
                           spamMetric='tracedist', method='ls', verbosity=10, checkJac=True)


        #with self.assertRaises(ValueError):
        #    self.runSilent(pygsti.gaugeopt_to_target, gs_lgst_target, self.gateset,
        #                   gatesMetric='foobar', verbosity=10) #bad gatesMetric
        #
        #with self.assertRaises(ValueError):
        #    self.runSilent(pygsti.gaugeopt_to_target, gs_lgst_target, self.gateset,
        #                   spamMetric='foobar', verbosity=10) #bad spamMetric
                

        #Contractions
        gs_clgst_tp    = self.runSilent(pygsti.contract, gs_lgst_target, "TP",verbosity=10, tol=10.0)
        gs_clgst_cp    = self.runSilent(pygsti.contract, gs_lgst_target, "CP",verbosity=10, tol=10.0)
        gs_clgst_cp2    = self.runSilent(pygsti.contract, gs_lgst_target, "CP",verbosity=10, tol=10.0)
        gs_clgst_cptp  = self.runSilent(pygsti.contract, gs_lgst_target, "CPTP",verbosity=10, tol=10.0)
        gs_clgst_cptp2 = self.runSilent(pygsti.contract, gs_lgst_target, "CPTP",verbosity=10, useDirectCP=False)
        gs_clgst_cptp3 = self.runSilent(pygsti.contract, gs_lgst_target, "CPTP",verbosity=10, tol=10.0, maxiter=0)
        gs_clgst_xp    = self.runSilent(pygsti.contract, gs_lgst_target, "XP", ds,verbosity=10, tol=10.0)
        gs_clgst_xptp  = self.runSilent(pygsti.contract, gs_lgst_target, "XPTP", ds,verbosity=10, tol=10.0)
        gs_clgst_vsp   = self.runSilent(pygsti.contract, gs_lgst_target, "vSPAM",verbosity=10, tol=10.0)
        gs_clgst_none  = self.runSilent(pygsti.contract, gs_lgst_target, "nothing",verbosity=10, tol=10.0)

          #test bad effect vector cases
        gs_bad_effect = gs_lgst_target.copy()
        gs_bad_effect.povms['Mdefault'] = pygsti.obj.UnconstrainedPOVM( [('0',[100.0,0,0,0])] ) # E eigvals all > 1.0
        self.runSilent(pygsti.contract, gs_bad_effect, "vSPAM",verbosity=10, tol=10.0)
        gs_bad_effect.povms['Mdefault'] = pygsti.obj.UnconstrainedPOVM( [('0',[-100.0,0,0,0])] ) # E eigvals all < 0
        self.runSilent(pygsti.contract, gs_bad_effect, "vSPAM",verbosity=10, tol=10.0)

        #with self.assertRaises(ValueError):
        #    self.runSilent(pygsti.contract, gs_lgst_target, "foobar",verbosity=10, tol=10.0) #bad toWhat

            
        #More gauge optimizations
        TP_gauge_group = pygsti.obj.TPGaugeGroup(gs_lgst.dim)
        gs_lgst_target_cp  = self.runSilent(pygsti.gaugeopt_to_target, gs_clgst_cptp, self.gateset, 
                                            cptp_penalty_factor=1.0, gauge_group=TP_gauge_group, verbosity=10, checkJac=True)

        gs_lgst_tp         = self.runSilent(pygsti.gaugeopt_to_target, gs_lgst, None,
                                                        spam_penalty_factor=1.0, verbosity=10, checkJac=True)

        gs_lgst.basis = Basis("gm",2) #so CPTP optimizations can work on gs_lgst
        gs_lgst_cptp       = self.runSilent(pygsti.gaugeopt_to_target, gs_lgst, None,
                                            cptp_penalty_factor=1.0, spam_penalty_factor=1.0, verbosity=10, checkJac=True)

        gs_lgst_cptp_tp    = self.runSilent(pygsti.gaugeopt_to_target, gs_lgst, None,
                                            cptp_penalty_factor=1.0, spam_penalty_factor=1.0, gauge_group=TP_gauge_group, verbosity=10, checkJac=True) #no point? (remove?)

        #I'm not sure why moving this test upward fixes a singlar matrix error (TODO LATER? - could one of above tests modify gs_lgst??)
        #gs_lgst_tp         = self.runSilent(pygsti.gaugeopt_to_target( gs_lgst, None,
        #                                    spam_penalty_factor=1.0, verbosity=10, checkJac=True)

        gs_lgst_tptarget   = self.runSilent(pygsti.gaugeopt_to_target, gs_lgst, self.gateset,
                                            spam_penalty_factor=1.0, verbosity=10, checkJac=True)

        gs_lgst_cptptarget = self.runSilent(pygsti.gaugeopt_to_target, gs_lgst, self.gateset,
                                            cptp_penalty_factor=1.0, spam_penalty_factor=1.0, verbosity=10, checkJac=True)

        gs_lgst_cptptarget2= self.runSilent(pygsti.gaugeopt_to_target, gs_lgst, self.gateset,
                                            cptp_penalty_factor=1.0, spam_penalty_factor=1.0, gauge_group=TP_gauge_group, verbosity=10, checkJac=True) #no point? (remove?)

        #Use "None" gauge group
        gs_none = self.runSilent(pygsti.gaugeopt_to_target, gs_lgst, self.gateset, gauge_group=None, verbosity=10)
        soln, trivialEl, gs_none = self.runSilent(pygsti.gaugeopt_to_target, gs_lgst, self.gateset, gauge_group=None, verbosity=10, returnAll=True)

        #Use "None" default gauge group
        gs_none = self.gateset.copy()
        gs_none.default_gauge_group = None
        self.runSilent(pygsti.gaugeopt_to_target, gs_none, self.gateset, verbosity=10)
        soln, trivialEl, gs_none = self.runSilent(pygsti.gaugeopt_to_target, gs_none, self.gateset, verbosity=10, returnAll=True)


        #TODO: check output lies in space desired

        # big kick that should land it outside XP, TP, etc, so contraction
        # routines are more tested
        gs_bigkick = gs_lgst_target.kick(absmag=1.0)
        gs_badspam = gs_bigkick.copy()
        gs_badspam.povms['Mdefault'] = pygsti.obj.UnconstrainedPOVM( [('0',np.array( [[2],[0],[0],[4]], 'd'))] )
          #set a bad evec so vSPAM has to work...
        

        gs_clgst_tp    = self.runSilent(pygsti.contract,gs_bigkick, "TP", verbosity=10, tol=10.0)
        gs_clgst_cp    = self.runSilent(pygsti.contract,gs_bigkick, "CP", verbosity=10, tol=10.0)
        gs_clgst_cptp  = self.runSilent(pygsti.contract,gs_bigkick, "CPTP", verbosity=10, tol=10.0)
        gs_clgst_xp    = self.runSilent(pygsti.contract,gs_bigkick, "XP", ds, verbosity=10, tol=10.0)
        gs_clgst_xptp  = self.runSilent(pygsti.contract,gs_bigkick, "XPTP", ds, verbosity=10, tol=10.0)
        gs_clgst_vsp   = self.runSilent(pygsti.contract,gs_badspam, "vSPAM", verbosity=10, tol=10.0)
        gs_clgst_none  = self.runSilent(pygsti.contract,gs_bigkick, "nothing", verbosity=10, tol=10.0)

        #TODO: check output lies in space desired

        #Check Errors
        with self.assertRaises(ValueError):
            pygsti.contract(gs_lgst_target, "FooBar",verbosity=0) # bad toWhat argument

        # No longer raise value error for failure to contract...
        #with self.assertRaises(ValueError):
        #    self.runSilent(pygsti.contract,gs_bigkick, "CP", verbosity=10,
        #                   maxiter=1) # fail to contract to CP


if __name__ == "__main__":
    unittest.main(verbosity=2)
