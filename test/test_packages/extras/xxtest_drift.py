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

    def test_drift_characterization(self):
        tds = pygsti.io.load_tddataset(compare_files + "/timeseries_data_trunc.txt")
        results_gst = drift.do_basic_drift_characterization(tds)
        results_gst.any_drift_detect()
        gstr = list(tds.keys())[0]
        
        print(results_gst.global_drift_frequencies)

        if bMPL:
            results_gst.plot_power_spectrum()
            results_gst.plot_power_spectrum(sequence=gstr,loc='upper right')

            # This box constructs some GST objects, needed to create any sort of boxplot with GST data

        # This manually specifies the germ and fiducial structure for the imported data.
        fiducial_strs = ['{}','Gx','Gy','GxGx','GxGxGx','GyGyGy']
        germ_strs = ['Gi','Gx','Gy','GxGy','GxGyGi','GxGiGy','GxGiGi','GyGiGi','GxGxGiGy','GxGyGyGi','GxGxGyGxGyGy']
        log2maxL = 1 # log2 of the maximum germ power

        # Below we use the maxlength, germ and fuducial lists to create the GST structures needed for box plots.
        fiducials = [pygsti.objects.GateString(None,fs) for fs in fiducial_strs]
        germs = [pygsti.objects.GateString(None,gs) for gs in germ_strs]
        max_lengths = [2**i for i in range(0,log2maxL)]
        gssList = pygsti.construction.make_lsgst_structs(std1Q_XYI.gates, fiducials, fiducials, germs, max_lengths)

        w = pygsti.report.Workspace()
        w.init_notebook_mode(connected=False, autodisplay=True)

        # Create a boxplot of the maximum power in the power spectra for each sequence.
        w.ColorBoxPlot('driftpwr', gssList[-1], None, None, driftresults = results_gst)

        if bMPL:
            results_gst.plot_most_drifty_probability(plot_data=True)

            gstrs = [pygsti.objects.GateString(None,'(Gx)^'+str(2**l)) for l in range(0,1)]
            results_gst.plot_multi_estimated_probabilities(gstrs,loc='upper right')


        #More from bottom of tutorial:
        fname = compare_files + "/timeseries_data_trunc.txt"
        data_gst, indices_to_gstrs = drift.load_bitstring_data(fname)
        gstrs_to_indices = results_gst.sequences_to_indices
        parray_gst = drift.load_bitstring_probabilities(fname,gstrs_to_indices)
        #EGN: Not sure why this final plot line doesn't work -- shape mismatch...
        #if bMPL:
        #    results_gst.plot_estimated_probability(sequence=pygsti.objects.GateString(None,'Gx(Gx)^1Gx'),plot_data=False,parray=parray_gst)


    def test_single_sequence_1Q(self):
            
        N = 5 # Counts per timestep
        T = 100 # Number of timesteps
        
        # The confidence of the statistical tests. Here we set it to 0.999, which means that
        # if we detect drift we are 0.999 confident that we haven't incorrectly rejected the
        # initial hypothesis of no drift.
        confidence = 0.999
        
        # A drifting probability to obtain the measurement outcome with index 1 (out of [0,1])
        def pt_drift(t): return 0.5+0.2*np.cos(0.1*t)
        
        # A drift-free probability to obtain the measurement outcome with index 1 (out of [0,1])
        def pt_nodrift(t): return 0.5
        
        # If we want the sequence to have a label, we define a list for this (here, a list of length 1).
        # The labels can, but need not be, pyGSTi GateString objects.
        sequences = [pygsti.objects.GateString(None,'Gx(Gi)^64Gx'),]
        
        # If we want the outcomes to have labels, we define a list for this.
        outcomes = ['0','1']
        
        # Let's create some fake data by sampling from these p(t) at integer times. Here we have
        # created a 1D array, but we could have instead created a 1 x 1 x 1 x T array.
        data_1seq_drift = np.array([binomial(N,pt_drift(t)) for t in range(0,T)])
        data_1seq_nodrift = np.array([binomial(N,pt_nodrift(t)) for t in range(0,T)])
        
        # If we want frequencies in Hertz, we need to specify the timestep in seconds. If this isn't
        # specified, the frequencies are given in 1/timestep with timestep defaulting to 1.
        timestep = 1e-5
        
        # We hand these 1D arrays to the analysis function, along with the number of counts, and other
        # optional information
        results_1seq_drift = drift.do_basic_drift_characterization(data_1seq_drift, counts=N, outcomes=outcomes,
                                                                   confidence=confidence, timestep=timestep, 
                                                                   indices_to_sequences=sequences)
        results_1seq_nodrift = drift.do_basic_drift_characterization(data_1seq_nodrift, counts=N, outcomes=outcomes, 
                                                                     confidence=confidence, timestep=timestep, 
                                                                     indices_to_sequences=sequences)

        if bMPL:
            results_1seq_drift.plot_power_spectrum()
            results_1seq_nodrift.plot_power_spectrum()

        print(results_1seq_drift.global_pvalue)
        print(results_1seq_nodrift.global_pvalue)

        # The power spectrum obtained after averaging over everthing
        print(results_1seq_drift.global_power_spectrum[:4])
        # The power spectrum obtained after averaging over everthing except sequence label
        print(results_1seq_drift.ps_power_spectrum[0,:4])
        # The power spectrum obtained after averaging over everthing except entity label
        print(results_1seq_drift.pe_power_spectrum[0,:4])
        # The power spectrum obtained after averaging over everthing except sequene and entity label
        print(results_1seq_drift.pspe_power_spectrum[0,0,:4])
        # The two power spectra obtained after averaging over nothing
        print(results_1seq_drift.pspepo_power_spectrum[0,0,0,:4])
        print(results_1seq_drift.pspepo_power_spectrum[0,0,1,:4])

        # Lets create an array of the true probability. This needs to be
        # of dimension S x E x M x T
        parray_1seq = np.zeros((1,1,2,T),float)
        parray_1seq[0,0,0,:] = np.array([pt_drift(t) for t in range(0,T)])
        parray_1seq[0,0,1,:] = 1 - parray_1seq[0,0,0,:]
        
        # The measurement outcome index we want to look at (here the esimated p(t) 
        # for one index is just 1 - the p(t) for the other index, because we are
        # looking at a two-outcome measurement).
        outcome = 1
        
        # If we hand the parray to the plotting function, it will also plot
        # the true probability alongside our estimate from the data
        if bMPL:
            results_1seq_drift.plot_estimated_probability(sequence=0,outcome=outcome,parray=parray_1seq,plot_data=True)

    def test_single_sequence_multiQ(self):
        outcomes = ['00','01','10','11']

        N = 10 # Counts per timestep
        T = 1000 # Number of timesteps
        
        # The drifting probabilities for the 4 outcomes
        def pt00(t): return (0.5+0.07*np.cos(0.08*t))*(0.5+0.08*np.cos(0.2*t))
        def pt01(t): return (0.5+0.07*np.cos(0.08*t))*(0.5-0.08*np.cos(0.2*t))
        def pt10(t): return (0.5-0.07*np.cos(0.08*t))*(0.5+0.08*np.cos(0.2*t))
        def pt11(t): return (0.5-0.07*np.cos(0.08*t))*(0.5-0.08*np.cos(0.2*t))
        
        # Because of the type of input (>2 measurement outcomes), we must record the
        # data in a 4D array (even though some of the dimensions are trivial)
        data_multiqubit = np.zeros((1,1,4,T),float)
        
        # Generate data from these p(t)
        for t in range(0,T):
            data_multiqubit[0,0,:,t] = multinomial(N,[pt00(t),pt01(t),pt10(t),pt11(t)])

        results_multiqubit_full = drift.do_basic_drift_characterization(data_multiqubit,outcomes=outcomes,confidence=0.99)
        print(results_multiqubit_full.global_drift_frequencies)

        if bMPL:
            results_multiqubit_full.plot_power_spectrum()
            outcome = 0 # the outcome index associated with the '00' outcome
            results_multiqubit_full.plot_power_spectrum(sequence=0,entity=0,outcome=outcome)

        print(results_multiqubit_full.pspepo_drift_frequencies[0,0,0])
        print(results_multiqubit_full.pspepo_drift_frequencies[0,0,1])
        print(results_multiqubit_full.pspepo_drift_frequencies[0,0,2])
        print(results_multiqubit_full.pspepo_drift_frequencies[0,0,3])

        # Creates an array of the true probability.
        parray_multiqubit_full = np.zeros((1,1,4,T),float)
        parray_multiqubit_full[0,0,0,:] = np.array([pt00(t) for t in range(0,T)])
        parray_multiqubit_full[0,0,1,:] = np.array([pt01(t) for t in range(0,T)])
        parray_multiqubit_full[0,0,2,:] = np.array([pt10(t) for t in range(0,T)])
        parray_multiqubit_full[0,0,3,:] = np.array([pt11(t) for t in range(0,T)])

        if bMPL:
            results_multiqubit_full.plot_estimated_probability(sequence=0,outcome=1, plot_data=True,
                                                               parray=parray_multiqubit_full)

        results_multiqubit_marg = drift.do_basic_drift_characterization(data_multiqubit, outcomes=outcomes, 
                                                                        marginalize = 'std', confidence=0.99)
        self.assertEqual(np.shape(results_multiqubit_marg.data), (1, 2, 2, 1000))

        if bMPL:
            results_multiqubit_marg.plot_power_spectrum()
            results_multiqubit_marg.plot_power_spectrum(sequence=0,entity=1)
            results_multiqubit_marg.plot_power_spectrum(sequence=0,entity=0)
            
        # Drift frequencies for the first qubit
        print(results_multiqubit_marg.pe_drift_frequencies[0])
        # Drift frequencies for the second qubit
        print(results_multiqubit_marg.pe_drift_frequencies[1])

        # Creates an array of the true probability.
        parray_multiqubit_marg = np.zeros((1,2,2,T),float)
        parray_multiqubit_marg[0,0,0,:] = np.array([pt00(t)+pt01(t) for t in range(0,T)])
        parray_multiqubit_marg[0,0,1,:] = np.array([pt10(t)+pt11(t) for t in range(0,T)])
        parray_multiqubit_marg[0,1,0,:] = np.array([pt00(t)+pt10(t) for t in range(0,T)])
        parray_multiqubit_marg[0,1,1,:] = np.array([pt01(t)+pt11(t) for t in range(0,T)])

        if bMPL:
            results_multiqubit_marg.plot_estimated_probability(sequence=0,entity=0,outcome=0,parray=parray_multiqubit_marg)


        N = 10 # Counts per timestep
        T = 1000 # Number of timesteps

        outcomes = ['00','01','10','11']
        
        def pt_correlated00(t): return 0.25-0.05*np.cos(0.05*t)
        def pt_correlated01(t): return 0.25+0.05*np.cos(0.05*t)
        def pt_correlated10(t): return 0.25+0.05*np.cos(0.05*t)
        def pt_correlated11(t): return 0.25-0.05*np.cos(0.05*t)
        
        data_1seq_multiqubit = np.zeros((1,1,4,T),float)
        for t in range(0,T):
            pvec = [pt_correlated00(t),pt_correlated01(t),pt_correlated10(t),pt_correlated11(t)]
            data_1seq_multiqubit[0,0,:,t] = multinomial(N,pvec)
        
        results_correlatedrift_marg = drift.do_basic_drift_characterization(data_1seq_multiqubit,
                                                                            outcomes=outcomes, marginalize = 'std')

        results_correlatedrift_full = drift.do_basic_drift_characterization(data_1seq_multiqubit, 
                                                                            outcomes=outcomes, marginalize = 'none')

        if bMPL:
            results_correlatedrift_marg.plot_power_spectrum()
            results_correlatedrift_full.plot_power_spectrum()

        

if __name__ == '__main__':
    unittest.main(verbosity=2)



