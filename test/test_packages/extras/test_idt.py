import pickle
import time
import unittest
import warnings

import numpy as np

import pygsti
import pygsti.models.modelconstruction as mc
from pygsti.processors.processorspec import QubitProcessorSpec as _ProcessorSpec
from pygsti.extras import idletomography as idt
from ..testutils import BaseTestCase, compare_files, temp_files, regenerate_references

#Helper functions
#Global dicts describing how to prep and measure in various bases
prepDict = { 'X': ('Gy',), 'Y': ('Gx',)*3, 'Z': (),
             '-X': ('Gy',)*3, '-Y': ('Gx',), '-Z': ('Gx','Gx')}
measDict = { 'X': ('Gy',)*3, 'Y': ('Gx',), 'Z': (),
             '-X': ('Gy',), '-Y': ('Gx',)*3, '-Z': ('Gx','Gx')}

#Global switches for debugging
hamiltonian=True
stochastic=True
affine=True

#Mimics a function that used to be in pyGSTi, replaced with create_cloudnoise_model_from_hops_and_weights
def build_XYCNOT_cloudnoise_model(nQubits, geometry="line", cnot_edges=None,
                                  maxIdleWeight=1, maxSpamWeight=1, maxhops=0,
                                  extraWeight1Hops=0, extraGateWeight=0,
                                  sparse_lindblad_basis=False, sparse_lindblad_reps=False,
                                  roughNoise=None, simulator="matrix", parameterization="H+S",
                                  spamtype="lindblad", addIdleNoiseToAllGates=True,
                                  errcomp_type="gates", return_clouds=False, verbosity=0):
    availability = {}; nonstd_gate_unitaries = {}
    if cnot_edges is not None: availability['Gcnot'] = cnot_edges
    pspec = _ProcessorSpec(nQubits, ['Gx', 'Gy', 'Gcnot'], nonstd_gate_unitaries, availability, geometry)
    assert (spamtype == "lindblad")  # unused and should remove this arg, but should always be "lindblad"
    mdl = mc.create_cloud_crosstalk_model_from_hops_and_weights(
        pspec, None,
        maxIdleWeight, maxSpamWeight, maxhops,
        extraWeight1Hops, extraGateWeight,
        simulator, 'default', parameterization, parameterization,
        addIdleNoiseToAllGates,
        errcomp_type, True, True, verbosity)

    if return_clouds:
        # FUTURE - just return cloud *keys*? (operation label values are never used
        # downstream, but may still be useful for debugging, so keep for now)
        return mdl, mdl.clouds
    else:
        return mdl


def get_fileroot(nQubits, maxMaxLen, errMag, spamMag, nSamples, simulator, idleErrorInFiducials):
    return temp_files + "/idletomog_%dQ_maxLen%d_errMag%.5f_spamMag%.5f_%s_%s_%s" % \
            (nQubits,maxMaxLen,errMag,spamMag,
             "nosampleerr" if (nSamples == "inf") else ("%dsamples" % nSamples),
             simulator, 'idleErrInFids' if idleErrorInFiducials else 'noIdleErrInFids')

def make_idle_tomography_data(nQubits, maxLengths=(0,1,2,4), errMags=(0.01,0.001), spamMag=0,
                              nSamplesList=(100,'inf'), simulator="map", sparsereps=False):

    base_param = []
    if hamiltonian: base_param.append('H')
    if stochastic: base_param.append('S')
    if affine: base_param.append('A')
    base_param = '+'.join(base_param)
    parameterization = base_param+" terms" if isinstance(simulator, pygsti.objects.TermForwardSimulator) else base_param # "H+S+A"

    gateset_idleInFids = build_XYCNOT_cloudnoise_model(nQubits, "line", [], min(2,nQubits), 1,
                                      simulator=simulator, parameterization=parameterization,
                                      roughNoise=None, addIdleNoiseToAllGates=True,
                                      sparse_lindblad_basis=False, sparse_lindblad_reps=sparsereps)
    gateset_noIdleInFids = build_XYCNOT_cloudnoise_model(nQubits, "line", [], min(2,nQubits), 1,
                                      simulator=simulator, parameterization=parameterization,
                                      roughNoise=None, addIdleNoiseToAllGates=False,
                                      sparse_lindblad_basis=False, sparse_lindblad_reps=sparsereps)

    listOfExperiments = idt.make_idle_tomography_list(nQubits, maxLengths, (prepDict,measDict), maxweight=min(2,nQubits),
                    include_hamiltonian=hamiltonian, include_stochastic=stochastic, include_affine=affine)

    base_vec = None
    for errMag in errMags:

        #ky = 'A(Z%s)' % ('I'*(nQubits-1)); debug_errdict = {ky: 0.01 }
        #ky = 'A(ZZ%s)' % ('I'*(nQubits-2)); debug_errdict = {ky: 0.01 }
        debug_errdict = {}
        if base_vec is None:
            rand_vec = idt.set_idle_errors(nQubits, gateset_idleInFids, debug_errdict, rand_default=errMag,
                                        hamiltonian=hamiltonian, stochastic=stochastic, affine=affine)
            base_vec = rand_vec / errMag

        err_vec = base_vec * errMag # for different errMags just scale the *same* random rates
        idt.set_idle_errors(nQubits, gateset_idleInFids, debug_errdict, rand_default=err_vec,
                          hamiltonian=hamiltonian, stochastic=stochastic, affine=affine)
        idt.set_idle_errors(nQubits, gateset_noIdleInFids, debug_errdict, rand_default=err_vec,
                          hamiltonian=hamiltonian, stochastic=stochastic, affine=affine) # same errors for w/ and w/out idle fiducial error

        for nSamples in nSamplesList:
            if nSamples == 'inf':
                sampleError = 'none'; Nsamp = 100
            else:
                sampleError = 'multinomial'; Nsamp = nSamples

            ds_idleInFids = pygsti.data.simulate_data(
                                gateset_idleInFids, listOfExperiments, num_samples=Nsamp,
                                sample_error=sampleError, seed=8675309)
            fileroot = get_fileroot(nQubits, maxLengths[-1], errMag, spamMag, nSamples, simulator, True)
            pickle.dump(gateset_idleInFids, open("%s_gs.pkl" % fileroot, "wb"))
            pickle.dump(ds_idleInFids, open("%s_ds.pkl" % fileroot, "wb"))
            print("Wrote fileroot ",fileroot)

            ds_noIdleInFids = pygsti.data.simulate_data(
                                gateset_noIdleInFids, listOfExperiments, num_samples=Nsamp,
                                sample_error=sampleError, seed=8675309)

            fileroot = get_fileroot(nQubits, maxLengths[-1], errMag, spamMag, nSamples, simulator, False)
            pickle.dump(gateset_noIdleInFids, open("%s_gs.pkl" % fileroot, "wb"))
            pickle.dump(ds_noIdleInFids, open("%s_ds.pkl" % fileroot, "wb"))

            #FROM DEBUGGING Python2 vs Python3 issue (ended up being an ordered-dict)
            ##pygsti.io.write_dataset("%s_ds_chk.txt" % fileroot, ds_noIdleInFids)
            #chk = pygsti.io.load_dataset("%s_ds_chk.txt" % fileroot)
            #for opstr,dsrow in ds_noIdleInFids.items():
            #    for outcome in dsrow.counts:
            #        cnt1, cnt2 = dsrow.counts.get(outcome,0.0),chk[opstr].counts.get(outcome,0.0)
            #        if not np.isclose(cnt1,cnt2):
            #            raise ValueError("NOT EQUAL: %s != %s" % (str(dsrow.counts), str(chk[opstr].counts)))
            #print("EQUAL!")

            print("Wrote fileroot ",fileroot)

def helper_idle_tomography(nQubits, maxLengths=(1,2,4), file_maxLen=4, errMag=0.01, spamMag=0, nSamples=100,
                         simulator="map", idleErrorInFiducials=True, fitOrder=1, fileroot=None):
    if fileroot is None:
        fileroot = get_fileroot(nQubits, file_maxLen, errMag, spamMag, nSamples, simulator, idleErrorInFiducials)

    mdl_datagen = pickle.load(open("%s_gs.pkl" % fileroot, "rb"))
    ds = pickle.load(open("%s_ds.pkl" % fileroot, "rb"))

    #print("DB: ",ds[ ('Gi',) ])
    #print("DB: ",ds[ ('Gi','Gi') ])
    #print("DB: ",ds[ ((('Gx',0),('Gx',1)),(('Gx',0),('Gx',1)),'Gi',(('Gx',0),('Gx',1)),(('Gx',0),('Gx',1))) ])

    advanced = {'fit order': fitOrder}
    results = idt.do_idle_tomography(nQubits, ds, maxLengths, (prepDict,measDict), maxweight=min(2,nQubits),
                                     advanced_options=advanced, include_hamiltonian=hamiltonian,
                                     include_stochastic=stochastic, include_affine=affine)

    if hamiltonian: ham_intrinsic_rates = results.intrinsic_rates['hamiltonian']
    if stochastic:  sto_intrinsic_rates = results.intrinsic_rates['stochastic']
    if affine:      aff_intrinsic_rates = results.intrinsic_rates['affine']

    maxErrWeight=2 # hardcoded for now
    datagen_ham_rates, datagen_sto_rates, datagen_aff_rates = \
        idt.predicted_intrinsic_rates(nQubits, maxErrWeight, mdl_datagen, hamiltonian, stochastic, affine)
    print("Predicted HAM = ",datagen_ham_rates)
    print("Predicted STO = ",datagen_sto_rates)
    print("Predicted AFF = ",datagen_aff_rates)
    print("Intrinsic HAM = ",ham_intrinsic_rates)
    print("Intrinsic STO = ",sto_intrinsic_rates)
    print("Intrinsic AFF = ",aff_intrinsic_rates)

    ham_diff = sto_diff = aff_diff = [0] # so max()=0 below for types we exclude
    if hamiltonian: ham_diff = np.abs(ham_intrinsic_rates - datagen_ham_rates)
    if stochastic:  sto_diff = np.abs(sto_intrinsic_rates - datagen_sto_rates)
    if affine:      aff_diff = np.abs(aff_intrinsic_rates - datagen_aff_rates)

    print("Err labels:", [ x.rep for x in results.error_list])
    if hamiltonian: print("Ham diffs:", ham_diff)
    if stochastic:  print("Sto diffs:", sto_diff)
    #if stochastic:
    #    for x,y in zip(sto_intrinsic_rates,datagen_sto_rates):
    #        print("  %g <--> %g" % (x,y))
    if affine:      print("Aff diffs:", aff_diff)
    print("%s\n MAX DIFFS: " % fileroot, max(ham_diff),max(sto_diff),max(aff_diff))
    return max(ham_diff),max(sto_diff),max(aff_diff)

#OLD - leftover from when we put data into a pandas data frame
#     #add hamiltonian data to df
#     N = len(labels) # number of hamiltonian/stochastic rates
#     data = pd.DataFrame({'nQubits': [nQubits]*N, 'maxL':[maxLengths[-1]]*N,
#             'errMag': [errMag]*N, 'spamMag': [spamMag]*N,
#             'nSamples': [nSamples]*N,
#             'simtype': [simtype]*N, 'type': ['hamiltonian']*N,
#             'true_val': datagen_ham_rates, 'estimate': ham_intrinsic_rates,
#             'diff': ham_intrinsic_rates - datagen_ham_rates, 'abs_diff': ham_diff,
#             'fitOrder': [fitOrder]*N, 'idleErrorInFiducials': [idleErrorInFiducials]*N })
#     df = df.append(data, ignore_index=True)

#     #add stochastic data to df
#     data = pd.DataFrame({'nQubits': [nQubits]*N, 'maxL':[maxLengths[-1]]*N,
#             'errMag': [errMag]*N, 'spamMag': [spamMag]*N,
#             'nSamples': [nSamples]*N,
#             'simtype': [simtype]*N, 'type': ['stochastic']*N,
#             'true_val': datagen_sto_rates, 'estimate': sto_intrinsic_rates,
#             'diff': sto_intrinsic_rates - datagen_sto_rates,'abs_diff': sto_diff,
#             'fitOrder': [fitOrder]*N, 'idleErrorInFiducials': [idleErrorInFiducials]*N })
#     df = df.append(data, ignore_index=True)
#     return df


class IDTTestCase(BaseTestCase):

    def test_idletomography_1Q(self):
        nQ = 1

        #make perfect data - using termorder:1 here means the data is not CPTP and
        # therefore won't be in [0,1], and creating a data set with sampleError="none"
        # means that probabilities *won't* be clipped to [0,1] - so we get really
        # funky and unphysical data here, but data that idle tomography should be
        # able to fit *exactly* (with any errMags, so be pick a big one).
        termsim = pygsti.objects.TermForwardSimulator(mode='taylor-order', max_order=1)
        make_idle_tomography_data(nQ, maxLengths=(0,1,2,4), errMags=(0.01,), spamMag=0,
                                  nSamplesList=('inf',), simulator=termsim, sparsereps=True)  # how specify order

        # Note: no spam error, as accounting for this isn't build into idle tomography yet.
        maxH, maxS, maxA = helper_idle_tomography(nQ, maxLengths=(1,2,4), file_maxLen=4,
                                                  errMag=0.01, spamMag=0, nSamples='inf',
                                                  idleErrorInFiducials=False, fitOrder=1, simulator=termsim)  # how specify order

        #Make sure exact identification of errors was possible
        self.assertLess(maxH, 1e-6)
        self.assertLess(maxS, 1e-6)
        self.assertLess(maxA, 1e-6)

    def test_idletomography_2Q(self):
        #Same thing but for 2 qubits
        nQ = 2
        termsim = pygsti.objects.TermForwardSimulator(mode='taylor-order', max_order=1)
        make_idle_tomography_data(nQ, maxLengths=(0,1,2,4), errMags=(0.01,), spamMag=0,
                                  nSamplesList=('inf',), simulator=termsim, sparsereps=True)  #How specify order?
        maxH, maxS, maxA = helper_idle_tomography(nQ, maxLengths=(1,2,4), file_maxLen=4,
                                                errMag=0.01, spamMag=0, nSamples='inf',
                                                  idleErrorInFiducials=False, fitOrder=1, simulator=termsim)  # how specify order?
        self.assertLess(maxH, 1e-6)
        self.assertLess(maxS, 1e-6)
        self.assertLess(maxA, 1e-6)

    def test_idletomog_gstdata_std1Q(self):
        from pygsti.modelpacks.legacy import std1Q_XYI as std
        std = pygsti.modelpacks.stdmodule_to_smqmodule(std)

        maxLens = [1,2,4]
        expList = pygsti.circuits.create_lsgst_circuits(std.target_model(), std.prepStrs,
                                                            std.effectStrs, std.germs_lite, maxLens)
        ds = pygsti.data.simulate_data(std.target_model().depolarize(0.01, 0.01),
                                               expList, 1000, 'multinomial', seed=1234)

        result = pygsti.run_long_sequence_gst(ds, std.target_model(), std.prepStrs, std.effectStrs, std.germs_lite, maxLens, verbosity=3)

        #standard report will run idle tomography
        pygsti.report.create_standard_report(result, temp_files + "/gstWithIdleTomogTestReportStd1Q",
                                             "Test GST Report w/Idle Tomography Tab: StdXYI",
                                             verbosity=3, auto_open=False)

    def test_idletomog_gstdata_1Qofstd2Q(self):
        # perform idle tomography on first qubit of 2Q
        from pygsti.modelpacks.legacy import std2Q_XYICNOT as std2Q
        from pygsti.modelpacks.legacy import std1Q_XYI as std
        std2Q = pygsti.modelpacks.stdmodule_to_smqmodule(std2Q)
        std = pygsti.modelpacks.stdmodule_to_smqmodule(std)

        maxLens = [1,2,4]
        expList = pygsti.circuits.create_lsgst_circuits(std2Q.target_model(), std2Q.prepStrs,
                                                            std2Q.effectStrs, std2Q.germs_lite, maxLens)
        mdl_datagen = std2Q.target_model().depolarize(0.01, 0.01)
        ds2Q = pygsti.data.simulate_data(mdl_datagen, expList, 1000, 'multinomial', seed=1234)

        #Just analyze first qubit (qubit 0)
        ds = pygsti.data.filter_dataset(ds2Q, (0,))

        start = std.target_model()
        start.set_all_parameterizations("TP")
        result = pygsti.run_long_sequence_gst(ds, start, std.prepStrs[0:4], std.effectStrs[0:4],
                                              std.germs_lite, maxLens, verbosity=3, advanced_options={'objective': 'chi2'})
        #result = pygsti.run_model_test(start.depolarize(0.009,0.009), ds, std.target_model(), std.prepStrs[0:4],
        #                              std.effectStrs[0:4], std.germs_lite, maxLens)
        pygsti.report.create_standard_report(result, temp_files + "/gstWithIdleTomogTestReportStd1Qfrom2Q",
                                             "Test GST Report w/Idle Tomog.: StdXYI from StdXYICNOT",
                                             verbosity=3, auto_open=False)

    def test_idletomog_gstdata_nQ(self):

        try: from pygsti.objects import fastreplib
        except ImportError:
            warnings.warn("Skipping test_idletomog_gstdata_nQ b/c no fastreps!")
            return


        #Global dicts describing how to prep and measure in various bases
        prepDict = { 'X': ('Gy',), 'Y': ('Gx',)*3, 'Z': (),
                     '-X': ('Gy',)*3, '-Y': ('Gx',), '-Z': ('Gx','Gx')}
        measDict = { 'X': ('Gy',)*3, 'Y': ('Gx',), 'Z': (),
                     '-X': ('Gy',), '-Y': ('Gx',)*3, '-Z': ('Gx','Gx')}

        nQubits = 2
        maxLengths = [1,2,4]

        ## ----- Generate n-qubit operation sequences -----
        if regenerate_references():
            c = {} #Uncomment to re-generate cache SAVE
        else:
            c = pickle.load(open(compare_files+"/idt_nQsequenceCache.pkl", 'rb'))

        t = time.time()
        gss = pygsti.circuits._create_xycnot_cloudnoise_circuits(
            nQubits, maxLengths, 'line', [(0,1)], max_idle_weight=2,
            idle_only=False, paramroot="H+S", cache=c, verbosity=3)
        #print("GSS STRINGS: ")
        #print('\n'.join(["%s: %s" % (s.str,str(s.tup)) for s in gss.allstrs]))

        gss_strs = gss.allstrs
        print("%.1fs" % (time.time()-t))
        if regenerate_references():
            pickle.dump(c, open(compare_files+"/idt_nQsequenceCache.pkl", 'wb'))
              #Uncomment to re-generate cache

        # To run idle tomography, we need "pauli fiducial pairs", so
        #  get fiducial pairs for Gi germ from gss and convert
        #  to "Pauli fidicual pairs" (which pauli state/basis is prepared or measured)
        GiStr = pygsti.obj.Circuit(((),), num_lines=nQubits)
        self.assertTrue(GiStr in gss.germs)
        self.assertTrue(gss.Ls == maxLengths)
        L0 = maxLengths[0] # all lengths should have same fidpairs, just take first one
        plaq = gss.get_plaquette(L0, GiStr)
        pauli_fidpairs = idt.fidpairs_to_pauli_fidpairs(plaq.fidpairs, (prepDict,measDict), nQubits)

        print(plaq.fidpairs)
        print()
        print('\n'.join([ "%s, %s" % (p[0],p[1]) for p in pauli_fidpairs]))
        self.assertEqual(len(plaq.fidpairs), len(pauli_fidpairs))
        self.assertEqual(len(plaq.fidpairs), 16) # (will need to change this if use H+S+A above)

        # ---- Create some fake data ----
        target_model = build_XYCNOT_cloudnoise_model(nQubits, "line", [(0,1)], 2, 1,
                                                     simulator="map", parameterization="H+S")

        #Note: generate data with affine errors too (H+S+A used below)
        mdl_datagen = build_XYCNOT_cloudnoise_model(nQubits, "line", [(0,1)], 2, 1,
                                                    simulator="map", parameterization="H+S+A",
                                                    roughNoise=(1234,0.001))
        #This *only* (re)sets Gi errors...
        idt.set_idle_errors(nQubits, mdl_datagen, {}, rand_default=0.001,
                  hamiltonian=True, stochastic=True, affine=True) # no seed? FUTURE?
        problemStr = pygsti.obj.Circuit([()], num_lines=nQubits)
        print("Problem: ",problemStr.str)
        assert(problemStr in gss.allstrs)
        ds = pygsti.data.simulate_data(mdl_datagen, gss.allstrs, 1000, 'multinomial', seed=1234)

        # ----- Run idle tomography with our custom (GST) set of pauli fiducial pairs ----
        advanced = {'pauli_fidpairs': pauli_fidpairs, 'jacobian mode': "together"}
        idtresults = idt.do_idle_tomography(nQubits, ds, maxLengths, (prepDict,measDict), maxweight=2,
                                     advanced_options=advanced, include_hamiltonian='auto',
                                     include_stochastic='auto', include_affine='auto')
        #Note: inclue_affine="auto" should have detected that we don't have the sequences to
        # determine the affine intrinsic rates:
        self.assertEqual(set(idtresults.intrinsic_rates.keys()), set(['hamiltonian','stochastic']))

        idt.create_idletomography_report(idtresults, temp_files + "/idleTomographyGSTSeqTestReport",
                                 "Test idle tomography report w/GST seqs", auto_open=False)


        #Run GST on the data (set tolerance high so this 2Q-GST run doesn't take long)
        gstresults = pygsti.run_long_sequence_gst_base(ds, target_model, gss,
                                                       advanced_options={'tolerance': 1e-1}, verbosity=3)

        #In FUTURE, we shouldn't need to set need to set the basis of our nQ GST results in order to make a report
        for estkey in gstresults.estimates: # 'default'
            gstresults.estimates[estkey].models['go0'].basis = pygsti.obj.Basis.cast("pp", 16)
            gstresults.estimates[estkey].models['target'].basis = pygsti.obj.Basis.cast("pp", 16)
        #pygsti.report.create_standard_report(gstresults, temp_files + "/gstWithIdleTomogTestReport",
        #                                    "Test GST Report w/Idle Tomography Tab",
        #                                    verbosity=3, auto_open=False)
        pygsti.report.create_nqnoise_report(gstresults, temp_files + "/gstWithIdleTomogTestReport",
                                            "Test nQNoise Report w/Idle Tomography Tab",
                                            verbosity=3, auto_open=False)


    def test_automatic_paulidicts(self):
        expected_prepDict = { 'X': ('Gy',), 'Y': ('Gx',)*3, 'Z': (),
                              '-X': ('Gy',)*3, '-Y': ('Gx',), '-Z': ('Gx','Gx')}
        expected_measDict = { 'X': ('Gy',)*3, 'Y': ('Gx',), 'Z': (),
                              '-X': ('Gy',), '-Y': ('Gx',)*3, '-Z': ('Gx','Gx')}

        target_model = build_XYCNOT_cloudnoise_model(3, "line", [(0,1)], 2, 1,
                                                     simulator="map", parameterization="H+S+A")
        prepDict, measDict = idt.determine_paulidicts(target_model)
        self.assertEqual(prepDict, expected_prepDict)
        self.assertEqual(measDict, expected_measDict)


if __name__ == '__main__':
    unittest.main(verbosity=2)
