import unittest

import numpy as np

import pygsti
from pygsti.extras import drift
from pygsti.modelpacks.legacy import std1Q_XYI
from ..testutils import BaseTestCase


class DriftTestCase(BaseTestCase):

    def test_signal(self):

        base = 0.5
        p = drift.signal.generate_gaussian_signal(10, 23, 10, 1000, base=base, method='sharp')
        p = drift.signal.generate_gaussian_signal(10, 23, 10, 1000, base=base, method='logistic')
        p = drift.signal.generate_gaussian_signal(10, 23, 10, 1000, base=base, method=None)
        p = drift.signal.generate_flat_signal(100, 2, 1000, candidatefreqs=None, base=0.5, method ='logistic')
        p = drift.signal.generate_flat_signal(100, 2, 1000, candidatefreqs=np.arange(1,10), base=0.5, method ='sharp')
        a1, b1, c1 = drift.signal.spectrum(p, times=np.arange(1000), returnfrequencies=True)
        a2, b2, c2 = drift.signal.spectrum(p, times=np.arange(1000), null_hypothesis=None, transform='lsp')
        a3, b3, c3 = drift.signal.spectrum(p, times=np.arange(1000), null_hypothesis=p, transform='dft')
        power = drift.signal.bartlett_spectrum(p, 5, counts=1, null_hypothesis=None, transform='dct')
        assert(abs(max(c1) - 50) < 1e-10)
        assert(abs(sum(c1[0:10]) - 100) < 1e-10)

        fnc = drift.signal.dct_basisfunction(3, np.arange(500), 0, 500)
        amps = drift.signal.amplitudes_at_frequencies([1, 3], {'0': fnc, '1': - fnc})

        assert(abs(amps['0'][0]) < 1e-10)
        assert(abs(amps['0'][1] - 1) < 1e-10)
        assert(abs(amps['1'][0]) < 1e-10)
        assert(abs(amps['1'][1] + 1) < 1e-10)

        powerthreshold = drift.signal.power_significance_threshold(0.05, 100, 1)
        pvaluethreshold = drift.signal.power_to_pvalue(powerthreshold, 1)
        assert(abs(pvaluethreshold - 0.05 / 100) < 1e-10)
        drift.signal.maxpower_pvalue(20, 100, 1)
        qthreshold = drift.signal.power_significance_quasithreshold(0.05, 100, 1, procedure='Benjamini-Hochberg')
        assert(abs(qthreshold[-1] - powerthreshold) < 1e-10)

        drift.signal.sparsity(p)
        plpf = drift.signal.lowpass_filter(p, max_freq=None)
        assert(np.sum(abs(p - plpf)) < 1e-7)

        assert(abs(drift.signal.moving_average(np.arange(0,100),width=11)[50] - 50) < 1e-10)

    def test_probtrajectory(self):

        # Create some fake clickstream data, from a constant probability distribution.
        numtimes = 500
        timstep = 0.1
        starttime = 0.
        times = (timstep * np.arange(0, numtimes)) + starttime
        clickstream = {}
        outcomes = ['0','1','2']
        for o in outcomes:
            clickstream[o] = []
        for i in range(len(times)):
            click = np.random.randint(0,3)
            for o in outcomes:
                if int(o) == click:
                    clickstream[o].append(1)
                else:
                    clickstream[o].append(0)

        # Test construction of a constant probability trajectory model
        pt = drift.probtrajectory.ConstantProbTrajectory(['0','1','2'],{'0':[0.5],'1':[0.2],})
        # Test MLE runs (this calls pretty much everything in the probability trajectory code)
        ptmax = drift.probtrajectory.maxlikelihood(pt, clickstream, times, verbosity=1)
        parameters = ptmax.get_parameters()
        # The exact MLE is the data mean, so check the returned MLE is close to that.
        for o in outcomes[:-1]:
            assert(abs(parameters[o][0] - np.mean(clickstream[o])) < 1e-3)
        # Check the minimization has actually increased the likelihood from the seed.
        assert(drift.probtrajectory.negloglikelihood(ptmax, clickstream, times) <= drift.probtrajectory.negloglikelihood(pt, clickstream, times))

        # Test construction of a DCT probability trajectory model
        ptdct = drift.probtrajectory.CosineProbTrajectory(['0','1','2'], [0,2], {'0':[0.5,0.02],'1':[0.2,0.03],}, 0, 0.1, 1000)
        # Test set parameters from list is working correctly.
        ptdct.set_parameters_from_list([0.5,0.02,0.2,0.03])
        assert(ptdct.get_parameters() == {'0':[0.5,0.02],'1':[0.2,0.03],})
        # Test set parameters from list is working correctly.
        assert(ptdct.get_parameters_as_list() == [0.5,0.02,0.2,0.03])
        # Run MLE.
        ptdctmax = drift.probtrajectory.maxlikelihood(ptdct, clickstream, times, verbosity=2)
        probsmax = ptdctmax.get_probabilities(times)
        # Check the minimization has actually increased the likelihood from the seed.
        assert(drift.probtrajectory.negloglikelihood(ptdctmax, clickstream, times) <= drift.probtrajectory.negloglikelihood(ptdct, clickstream, times))

        ptdct_invalid = drift.probtrajectory.CosineProbTrajectory(['0','1','2'], [0,2], {'0':[0.5, 0.5],'1':[0.2, 1.2],}, 0, 0.1, 1000)
        pt, check = drift.probtrajectory.amplitude_compression(ptdct_invalid, np.linspace(0,1000,2000))
        assert(check)
        params = pt.get_parameters()

    def test_timeresolvemodel(self):
    
        # A trmodel is a baseclass, and it's pretty trivial.
        trmodel = drift.trmodel.TimeResolvedModel([0,],[0.4])
        trmodel.get_parameters()
        trmodel.set_parameters([0.2])
        trmodel.hyperparameters

    @unittest.skip("Need to update this test - do_stability_analysis no longer exists")
    def test_core_and_stabilityanalyzer(self):

        ds = pygsti.io.load_time_dependent_dataset("cmp_chk_files/timeseries_data_trunc.txt")
        fiducial_strs = ['{}', 'Gx', 'Gy', 'GxGx', 'GxGxGx', 'GyGyGy']
        germ_strs = ['Gi']
        fiducials = [pygsti.objects.Circuit(None, stringrep=fs) for fs in fiducial_strs]
        germs = [pygsti.objects.Circuit(None, stringrep=mdl) for mdl in germ_strs]
        max_lengths = [256, ]
        gssList = pygsti.construction.make_lsgst_structs(std1Q_XYI.gates, fiducials, fiducials, germs, max_lengths)
        gss = gssList[-1]

        # Test the integrated routine
        results = drift.do_stability_analysis(ds, significance=0.01, estimator='mle')

        # Redo but testing direct access of the stability analyzer.
        results = drift.StabilityAnalyzer(ds, ids=True)
        print(results)
        results.compute_spectra()
        print(results)
        results.run_instability_detection(verbosity=0)
        print(results)
        results.run_instability_characterization(estimator='filter', modelselector=('default',()),verbosity=0)
        print(results)

        # Test adding more detectors.
        inclass_correction={'dataset': 'Bonferroni', 'circuit': 'Bonferroni', 'spectrum': 'Benjamini-Hochberg'}
        results.run_instability_detection(inclass_correction=inclass_correction, tests=(('circuit',),), saveas='fdr-1', 
                                         default=False, verbosity=0)
        inclass_correction={'dataset': 'Bonferroni', 'circuit': 'Benjamini-Hochberg', 'spectrum': 'Benjamini-Hochberg',}
        results.run_instability_detection(inclass_correction=inclass_correction, tests=((),('circuit',),), saveas='fdr-2',
                                         default=False)
        results.run_instability_detection(inclass_correction=inclass_correction, tests=(('circuit',),), saveas='fdr-3',
                                         default=False)

        results.get_unstable_circuits(fromtests=[(), ('circuit',)])

        freq, s = results.get_spectrum()
        circuit = ds.keys()[0]

        #Create a workspace to show plots
        w = pygsti.report.Workspace()
        #w.init_notebook_mode(connected=False, autodisplay=True) 
        w.ProbTrajectoriesPlot(results, circuit, outcome=('0',))
        w.PowerSpectraPlot(results, detectorkey='fdr-2')
        w.PowerSpectraPlot(results, spectrumlabel={'circuit': circuit})
        w.PowerSpectraPlot(results, spectrumlabel={'dataset': '0', 'circuit': [c for c in ds.keys()[:10]], 'outcome': ('1',)},
                           showlegend=True)
        w.ProbTrajectoriesPlot(results, [c for c in ds.keys()[:5]], outcome=('1', ))
        w.GermFiducialProbTrajectoriesPlot(results, gss, 'Gx', 'Gi', 'Gx', outcome=('1', ))
        w.GermFiducialPowerSpectraPlot(results, gss, 'Gx', 'Gi', 'Gx')

        drift.driftreport.create_drift_report(results, gss, "../temp_test_files/DriftReport", title="Test Drift Report",
                                              verbosity=10)


if __name__ == '__main__':
    unittest.main(verbosity=2)
