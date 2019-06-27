from ..testutils import BaseTestCase, compare_files, temp_files
import unittest
import numpy as np
from numpy.random import binomial
from numpy.random import multinomial
import os
import scipy
import pygsti

from pygsti.extras import drift
from pygsti.construction import std1Q_XYI
try:
    import matplotlib
    bMPL = True
except ImportError:
    bMPL = False

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

        ptdct_invalid = drift.probtrajectory.CosineProbTrajectory(['0','1','2'], [0,2], {'0':[0.5,0.5],'1':[0.2,0.5],}, 0, 0.1, 1000)
        pt, check = drift.probtrajectory.amplitude_compression(ptdct_invalid)
        assert(check)
        params = pt.get_parameters() 
        assert(np.allclose(params['0'] , np.array([0.5 , 0.15])))
        assert(np.allclose(params['1'] , np.array([0.2 , 0.15])))

    # def test_drift_characterization(self):
    #     tds = pygsti.io.load_tddataset(compare_files + "/timeseries_data_trunc.txt")
    #     results_gst = drift.do_drift_characterization(tds)
    #             #results_gst.any_drift_detect()
    #             # opstr = list(tds.keys())[0]
        
    #     # print(results_gst)

    #     #if bMPL:
    #     #    results_gst.plot_power_spectrum(savepath=temp_files+"/driftchar_powspec1.png")
    #     #    results_gst.plot_power_spectrum(sequence=opstr,loc='upper right', 
    #     #                                    savepath=temp_files+"/driftchar_powspec2.png")

    #         # This box constructs some GST objects, needed to create any sort of boxplot with GST data

    #     # This manually specifies the germ and fiducial structure for the imported data.
    #     fiducial_strs = ['{}','Gx','Gy','GxGx','GxGxGx','GyGyGy']
    #     germ_strs = ['Gi','Gx','Gy','GxGy','GxGyGi','GxGiGy','GxGiGi','GyGiGi','GxGxGiGy','GxGyGyGi','GxGxGyGxGyGy']
    #     log2maxL = 1 # log2 of the maximum germ power

    #     # Below we use the maxlength, germ and fuducial lists to create the GST structures needed for box plots.
    #     fiducials = [pygsti.objects.Circuit(None,stringrep=fs) for fs in fiducial_strs]
    #     germs = [pygsti.objects.Circuit(None,stringrep=s) for s in germ_strs]
    #     max_lengths = [2**i for i in range(0,log2maxL)]
    #     gssList = pygsti.construction.make_lsgst_structs(std1Q_XYI.gates, fiducials, fiducials, germs, max_lengths)

    #     w = pygsti.report.Workspace()
    #     # Create a boxplot of the maximum power in the power spectra for each sequence.
    #     w.ColorBoxPlot('driftpwr', gssList[-1], None, None, driftresults = (results_gst,None))

 







        #if bMPL:
        #    results_gst.plot_most_drifty_probability(plot_data=True, savepath=temp_files+"/driftchar_probs.png")
        #    
        #    gstrs = [pygsti.objects.Circuit(None,stringrep='(Gx)^'+str(2**l)) for l in range(0,1)]
        #    results_gst.plot_multi_estimated_probabilities(gstrs,loc='upper right',
        #                                                   savepath=temp_files+"/driftchar_multiprobs.png")


        #More from bottom of tutorial:
        #fname = compare_files + "/timeseries_data_trunc.txt"
        #data_gst, indices_to_gstrs = drift.load_bitstring_data(fname)
        #gstrs_to_indices = results_gst.sequences_to_indices
        #parray_gst = drift.load_bitstring_probabilities(fname,gstrs_to_indices)
        #EGN: Not sure why this final plot line doesn't work -- shape mismatch...
        #if bMPL:
        #    results_gst.plot_estimated_probability(sequence=pygsti.objects.Circuit(None,stringrep='Gx(Gx)^1Gx'),plot_data=False,parray=parray_gst)


    # def test_single_sequence_1Q(self):
            
    #     N = 5 # Counts per timestep
    #     T = 100 # Number of timesteps
        
    #     # The confidence of the statistical tests. Here we set it to 0.999, which means that
    #     # if we detect drift we are 0.999 confident that we haven't incorrectly rejected the
    #     # initial hypothesis of no drift.
    #     confidence = 0.999
        
    #     # A drifting probability to obtain the measurement outcome with index 1 (out of [0,1])
    #     def pt_drift(t): return 0.5+0.2*np.cos(0.1*t)
        
    #     # A drift-free probability to obtain the measurement outcome with index 1 (out of [0,1])
    #     def pt_nodrift(t): return 0.5
        
    #     # If we want the sequence to have a label, we define a list for this (here, a list of length 1).
    #     # The labels can, but need not be, pyGSTi Circuit objects.
    #     sequences = [pygsti.objects.Circuit(None,stringrep='Gx(Gi)^64Gx'),]
        
    #     # If we want the outcomes to have labels, we define a list for this.
    #     outcomes = ['0','1']
        
    #     # Let's create some fake data by sampling from these p(t) at integer times. Here we have
    #     # created a 1D array, but we could have instead created a 1 x 1 x 1 x T array.
    #     data_1seq_drift = np.array([binomial(N,pt_drift(t)) for t in range(0,T)])
    #     data_1seq_nodrift = np.array([binomial(N,pt_nodrift(t)) for t in range(0,T)])
        
    #     # If we want frequencies in Hertz, we need to specify the timestep in seconds. If this isn't
    #     # specified, the frequencies are given in 1/timestep with timestep defaulting to 1.
    #     timestep = 1e-5
        
    #     # We hand these 1D arrays to the analysis function, along with the number of counts, and other
    #     # optional information
    #     results_1seq_drift = drift.do_drift_characterization(data_1seq_drift, counts=N, outcomes=outcomes,
    #                                                                confidence=confidence, timestep=timestep, 
    #                                                                indices_to_sequences=sequences)
    #     results_1seq_nodrift = drift.do_drift_characterization(data_1seq_nodrift, counts=N, outcomes=outcomes, 
    #                                                                  confidence=confidence, timestep=timestep, 
    #                                                                  indices_to_sequences=sequences)

        # if bMPL:
        #     results_1seq_drift.plot_power_spectrum(savepath=temp_files+"/drift_powspec_1seqA.png")
        #     results_1seq_nodrift.plot_power_spectrum(savepath=temp_files+"/drift_powspec_1seqB.png")

        # print(results_1seq_drift.global_pvalue)
        # print(results_1seq_nodrift.global_pvalue)

        # # The power spectrum obtained after averaging over everthing
        # print(results_1seq_drift.global_power_spectrum[:4])
        # # The power spectrum obtained after averaging over everthing except sequence label
        # print(results_1seq_drift.ps_power_spectrum[0,:4])
        # # The power spectrum obtained after averaging over everthing except entity label
        # print(results_1seq_drift.pe_power_spectrum[0,:4])
        # # The power spectrum obtained after averaging over everthing except sequene and entity label
        # print(results_1seq_drift.pspe_power_spectrum[0,0,:4])
        # # The two power spectra obtained after averaging over nothing
        # print(results_1seq_drift.pspepo_power_spectrum[0,0,0,:4])
        # print(results_1seq_drift.pspepo_power_spectrum[0,0,1,:4])

        # # Lets create an array of the true probability. This needs to be
        # # of dimension S x E x M x T
        # parray_1seq = np.zeros((1,1,2,T),float)
        # parray_1seq[0,0,0,:] = np.array([pt_drift(t) for t in range(0,T)])
        # parray_1seq[0,0,1,:] = 1 - parray_1seq[0,0,0,:]
        
        # # The measurement outcome index we want to look at (here the esimated p(t) 
        # # for one index is just 1 - the p(t) for the other index, because we are
        # # looking at a two-outcome measurement).
        # outcome = 1
        
        # # If we hand the parray to the plotting function, it will also plot
        # # the true probability alongside our estimate from the data
        # if bMPL:
        #     results_1seq_drift.plot_estimated_probability(sequence=0,outcome=outcome,parray=parray_1seq,
        #                                                   plot_data=True, savepath=temp_files+"/drift_estprob1.png")

    # def test_single_sequence_multiQ(self):
    #     outcomes = ['00','01','10','11']

    #     N = 10 # Counts per timestep
    #     T = 1000 # Number of timesteps
        
    #     # The drifting probabilities for the 4 outcomes
    #     def pt00(t): return (0.5+0.07*np.cos(0.08*t))*(0.5+0.08*np.cos(0.2*t))
    #     def pt01(t): return (0.5+0.07*np.cos(0.08*t))*(0.5-0.08*np.cos(0.2*t))
    #     def pt10(t): return (0.5-0.07*np.cos(0.08*t))*(0.5+0.08*np.cos(0.2*t))
    #     def pt11(t): return (0.5-0.07*np.cos(0.08*t))*(0.5-0.08*np.cos(0.2*t))
        
    #     # Because of the type of input (>2 measurement outcomes), we must record the
    #     # data in a 4D array (even though some of the dimensions are trivial)
    #     data_multiqubit = np.zeros((1,1,4,T),float)
        
    #     # Generate data from these p(t)
    #     for t in range(0,T):
    #         data_multiqubit[0,0,:,t] = multinomial(N,[pt00(t),pt01(t),pt10(t),pt11(t)])

    #     results_multiqubit_full = drift.do_drift_characterization(data_multiqubit,outcomes=outcomes,confidence=0.99)
    #     print(results_multiqubit_full.global_drift_frequencies)

    #     if bMPL:
    #         results_multiqubit_full.plot_power_spectrum(savepath=temp_files+"/drift_powspec0.png")
    #         outcome = '00' 
    #         #OLD: outcome = 0 # the outcome index associated with the '00' outcome
    #         results_multiqubit_full.plot_power_spectrum(sequence=0,entity=0,outcome=outcome,
    #                                                     savepath=temp_files+"/drift_powspec1.png")

    #     print(results_multiqubit_full.pspepo_drift_frequencies[0,0,0])
    #     print(results_multiqubit_full.pspepo_drift_frequencies[0,0,1])
    #     print(results_multiqubit_full.pspepo_drift_frequencies[0,0,2])
    #     print(results_multiqubit_full.pspepo_drift_frequencies[0,0,3])

    #     # Creates an array of the true probability.
    #     parray_multiqubit_full = np.zeros((1,1,4,T),float)
    #     parray_multiqubit_full[0,0,0,:] = np.array([pt00(t) for t in range(0,T)])
    #     parray_multiqubit_full[0,0,1,:] = np.array([pt01(t) for t in range(0,T)])
    #     parray_multiqubit_full[0,0,2,:] = np.array([pt10(t) for t in range(0,T)])
    #     parray_multiqubit_full[0,0,3,:] = np.array([pt11(t) for t in range(0,T)])

    #     if bMPL:
    #         results_multiqubit_full.plot_estimated_probability(sequence=0,outcome=1, plot_data=True,
    #                                                            parray=parray_multiqubit_full,
    #                                                            savepath=temp_files+"/drift_estprob2.png")

    #     results_multiqubit_marg = drift.do_basic_drift_characterization(data_multiqubit, outcomes=outcomes, 
    #                                                                     marginalize = 'std', confidence=0.99)
    #     self.assertEqual(np.shape(results_multiqubit_marg.data), (1, 2, 2, 1000))

    #     if bMPL:
    #         results_multiqubit_marg.plot_power_spectrum(savepath=temp_files+"/drift_powspec2.png")
    #         results_multiqubit_marg.plot_power_spectrum(sequence=0,entity=1,savepath=temp_files+"/drift_powspec3.png")
    #         results_multiqubit_marg.plot_power_spectrum(sequence=0,entity=0,savepath=temp_files+"/drift_powspec4.png")
            
    #     # Drift frequencies for the first qubit
    #     print(results_multiqubit_marg.pe_drift_frequencies[0])
    #     # Drift frequencies for the second qubit
    #     print(results_multiqubit_marg.pe_drift_frequencies[1])

    #     # Creates an array of the true probability.
    #     parray_multiqubit_marg = np.zeros((1,2,2,T),float)
    #     parray_multiqubit_marg[0,0,0,:] = np.array([pt00(t)+pt01(t) for t in range(0,T)])
    #     parray_multiqubit_marg[0,0,1,:] = np.array([pt10(t)+pt11(t) for t in range(0,T)])
    #     parray_multiqubit_marg[0,1,0,:] = np.array([pt00(t)+pt10(t) for t in range(0,T)])
    #     parray_multiqubit_marg[0,1,1,:] = np.array([pt01(t)+pt11(t) for t in range(0,T)])

    #     if bMPL:
    #         results_multiqubit_marg.plot_estimated_probability(sequence=0,entity=0,outcome=0,
    #                                                            parray=parray_multiqubit_marg,
    #                                                            savepath=temp_files+"/drift_estprob3.png")


    #     N = 10 # Counts per timestep
    #     T = 1000 # Number of timesteps

    #     outcomes = ['00','01','10','11']
        
    #     def pt_correlated00(t): return 0.25-0.05*np.cos(0.05*t)
    #     def pt_correlated01(t): return 0.25+0.05*np.cos(0.05*t)
    #     def pt_correlated10(t): return 0.25+0.05*np.cos(0.05*t)
    #     def pt_correlated11(t): return 0.25-0.05*np.cos(0.05*t)
        
    #     data_1seq_multiqubit = np.zeros((1,1,4,T),float)
    #     for t in range(0,T):
    #         pvec = [pt_correlated00(t),pt_correlated01(t),pt_correlated10(t),pt_correlated11(t)]
    #         data_1seq_multiqubit[0,0,:,t] = multinomial(N,pvec)
        
    #     results_correlatedrift_marg = drift.do_basic_drift_characterization(data_1seq_multiqubit,
    #                                                                         outcomes=outcomes, marginalize = 'std')

    #     results_correlatedrift_full = drift.do_basic_drift_characterization(data_1seq_multiqubit, 
    #                                                                         outcomes=outcomes, marginalize = 'none')

    #     if bMPL:
    #         results_correlatedrift_marg.plot_power_spectrum(savepath=temp_files+"/drift_powspec5.png")
    #         results_correlatedrift_full.plot_power_spectrum(savepath=temp_files+"/drift_powspec6.png")

        

if __name__ == '__main__':
    unittest.main(verbosity=2)



