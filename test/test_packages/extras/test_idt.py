from ..testutils import BaseTestCase, compare_files, temp_files
import unittest
import numpy as np
import pickle

import pygsti
from pygsti.extras import idletomography as idt

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


def get_fileroot(nQubits, maxMaxLen, errMag, spamMag, nSamples, simtype, idleErrorInFiducials):
    return temp_files + "/idletomog_%dQ_maxLen%d_errMag%.5f_spamMag%.5f_%s_%s_%s" % \
            (nQubits,maxMaxLen,errMag,spamMag,
             "nosampleerr" if (nSamples == "inf") else ("%dsamples" % nSamples),
             simtype, 'idleErrInFids' if idleErrorInFiducials else 'noIdleErrInFids')

def make_idle_tomography_data(nQubits, maxLengths=(0,1,2,4), errMags=(0.01,0.001), spamMag=0,
                              nSamplesList=(100,'inf'), simtype="map"):
    
    prepNoise = (456,spamMag) if spamMag > 0 else None
    povmNoise = (789,spamMag) if spamMag > 0 else None
    base_param = []
    if hamiltonian: base_param.append('H')
    if stochastic: base_param.append('S')
    if affine: base_param.append('A')
    base_param = '+'.join(base_param)
    parameterization = base_param+" terms" if simtype.startswith('termorder') else base_param # "H+S+A"
    
    gateset_idleInFids = pygsti.construction.build_nqnoise_gateset(nQubits, "line", [], min(2,nQubits), 1,
                                      sim_type=simtype, parameterization=parameterization,
                                      gateNoise=None, prepNoise=prepNoise, povmNoise=povmNoise,
                                      addIdleNoiseToAllGates=True)
    gateset_noIdleInFids = pygsti.construction.build_nqnoise_gateset(nQubits, "line", [], min(2,nQubits), 1,
                                      sim_type=simtype, parameterization=parameterization,
                                      gateNoise=None, prepNoise=prepNoise, povmNoise=povmNoise,
                                      addIdleNoiseToAllGates=False)
    
    listOfExperiments = idt.make_idle_tomography_list(nQubits, (prepDict,measDict), maxLengths, maxErrWeight=min(2,nQubits),
                    includeHamSeqs=hamiltonian, includeStochasticSeqs=stochastic, includeAffineSeqs=affine)
    
    base_vec = None
    for errMag in errMags:
        
        #ky = 'A(Z%s)' % ('I'*(nQubits-1)); debug_errdict = {ky: 0.01 }
        #ky = 'A(ZZ%s)' % ('I'*(nQubits-2)); debug_errdict = {ky: 0.01 }
        debug_errdict = {}
        if base_vec is None:
            rand_vec = idt.set_Gi_errors(nQubits, gateset_idleInFids, debug_errdict, rand_default=errMag,
                                        hamiltonian=hamiltonian, stochastic=stochastic, affine=affine)
            base_vec = rand_vec / errMag
            
        err_vec = base_vec * errMag # for different errMags just scale the *same* random rates
        idt.set_Gi_errors(nQubits, gateset_idleInFids, debug_errdict, rand_default=err_vec,
                          hamiltonian=hamiltonian, stochastic=stochastic, affine=affine)
        idt.set_Gi_errors(nQubits, gateset_noIdleInFids, debug_errdict, rand_default=err_vec,
                          hamiltonian=hamiltonian, stochastic=stochastic, affine=affine) # same errors for w/ and w/out idle fiducial error
    
        for nSamples in nSamplesList:
            if nSamples == 'inf':
                sampleError = 'none'; Nsamp = 100
            else:
                sampleError = 'multinomial'; Nsamp = nSamples
                                
            ds_idleInFids = pygsti.construction.generate_fake_data(
                                gateset_idleInFids, listOfExperiments, nSamples=Nsamp,
                                sampleError=sampleError, seed=8675309)
            fileroot = get_fileroot(nQubits, maxLengths[-1], errMag, spamMag, nSamples, simtype, True)
            pickle.dump(gateset_idleInFids, open("%s_gs.pkl" % fileroot, "wb"))
            pickle.dump(ds_idleInFids, open("%s_ds.pkl" % fileroot, "wb"))
            print("Wrote fileroot ",fileroot)
            
            ds_noIdleInFids = pygsti.construction.generate_fake_data(
                                gateset_noIdleInFids, listOfExperiments, nSamples=Nsamp,
                                sampleError=sampleError, seed=8675309)
            fileroot = get_fileroot(nQubits, maxLengths[-1], errMag, spamMag, nSamples, simtype, False) 
            pickle.dump(gateset_noIdleInFids, open("%s_gs.pkl" % fileroot, "wb"))
            pickle.dump(ds_noIdleInFids, open("%s_ds.pkl" % fileroot, "wb"))
            print("Wrote fileroot ",fileroot)
            
def helper_idle_tomography(nQubits, maxLengths=(1,2,4), file_maxLen=4, errMag=0.01, spamMag=0, nSamples=100,
                         simtype="map", idleErrorInFiducials=True, fitOrder=1, fileroot=None):   
    if fileroot is None:
        fileroot = get_fileroot(nQubits, file_maxLen, errMag, spamMag, nSamples, simtype, idleErrorInFiducials) 
            
    gs_datagen = pickle.load(open("%s_gs.pkl" % fileroot, "rb"))
    ds = pickle.load(open("%s_ds.pkl" % fileroot, "rb"))
    
    #print("DB: ",ds[ ('Gi',) ])
    #print("DB: ",ds[ ('Gi','Gi') ])
    #print("DB: ",ds[ ((('Gx',0),('Gx',1)),(('Gx',0),('Gx',1)),'Gi',(('Gx',0),('Gx',1)),(('Gx',0),('Gx',1))) ])
    
    advanced = {'fit order': fitOrder}
    results = idt.do_idle_tomography(nQubits, ds, maxLengths, (prepDict,measDict), maxErrWeight=min(2,nQubits),
                                     advancedOptions=advanced, extract_hamiltonian=hamiltonian,
                                     extract_stochastic=stochastic, extract_affine=affine)
        
    if hamiltonian: ham_intrinsic_rates = results.intrinsic_rates['hamiltonian']
    if stochastic:  sto_intrinsic_rates = results.intrinsic_rates['stochastic'] 
    if affine:      aff_intrinsic_rates = results.intrinsic_rates['affine'] 
        
    maxErrWeight=2 # hardcoded for now
    datagen_ham_rates, datagen_sto_rates, datagen_aff_rates = \
        idt.predicted_intrinsic_rates(nQubits, maxErrWeight, gs_datagen, hamiltonian, stochastic, affine)
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
        make_idle_tomography_data(nQ, maxLengths=(0,1,2,4), errMags=(0.01,), spamMag=0,
                                  nSamplesList=('inf',), simtype="termorder:1")

        # Note: no spam error, as accounting for this isn't build into idle tomography yet.
        maxH, maxS, maxA = helper_idle_tomography(nQ, maxLengths=(1,2,4), file_maxLen=4,
                                                errMag=0.01, spamMag=0, nSamples='inf',
                                                idleErrorInFiducials=False, fitOrder=1, simtype="termorder:1")

        #Make sure exact identification of errors was possible
        self.assertLess(maxH, 1e-6)
        self.assertLess(maxS, 1e-6)
        self.assertLess(maxA, 1e-6)

    def test_idletomography_2Q(self):        
        #Same thing but for 2 qubits
        nQ = 2
        make_idle_tomography_data(nQ, maxLengths=(0,1,2,4), errMags=(0.01,), spamMag=0,
                                  nSamplesList=('inf',), simtype="termorder:1")
        maxH, maxS, maxA = helper_idle_tomography(nQ, maxLengths=(1,2,4), file_maxLen=4,
                                                errMag=0.01, spamMag=0, nSamples='inf',
                                                idleErrorInFiducials=False, fitOrder=1, simtype="termorder:1")
        self.assertLess(maxH, 1e-6)
        self.assertLess(maxS, 1e-6)
        self.assertLess(maxA, 1e-6)
        

if __name__ == '__main__':
    unittest.main(verbosity=2)



