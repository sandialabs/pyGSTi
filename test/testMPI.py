import unittest
import time
import sys
from mpinoseutils import *

import pygsti


bUseMPI = True

if bUseMPI: 
    from mpi4py import MPI
    g_comm = MPI.COMM_WORLD
    g_rank = g_comm.Get_rank()
else:
    g_comm = None
    g_rank = 0


class MPITestCase(unittest.TestCase):

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


class TestMPIMethods(MPITestCase):
    
    def runOneQubit_Tutorial(self):
        from pygsti.construction import std1Q_XYI
        gs_target = std1Q_XYI.gs_target
        fiducials = std1Q_XYI.fiducials
        germs = std1Q_XYI.germs
        maxLengths = [0,1,2,4,8,16,32,64,128,256,512,1024,2048]
        
        gs_datagen = gs_target.depolarize(gate_noise=0.1, spam_noise=0.001)
        listOfExperiments = pygsti.construction.make_lsgst_experiment_list(
            gs_target.gates.keys(), fiducials, fiducials, germs, maxLengths)
        ds = pygsti.construction.generate_fake_data(gs_datagen, listOfExperiments,
                                                    nSamples=1000,
                                                    sampleError="binomial",
                                                    seed=1234)
    
        results = pygsti.do_long_sequence_gst(ds, gs_target, fiducials, fiducials,
                                              germs, maxLengths, comm=comm)
        #results.create_full_report_pdf(confidenceLevel=95,filename="tutorial_files/MyEvenEasierReport.pdf",verbosity=2)
    
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


def runMC2GSTAnalysis(myspecs, mygerms, gsTarget, seed,
                      maxLs = [1,2,4,8],
                      nSamples=1000, useFreqWeightedChiSq=False,
                      minProbClipForWeighting=1e-4, fidPairList=None,
                      comm=None):
    rhoStrs, EStrs = pygsti.construction.get_spam_strs(myspecs)
    lgstStrings = pygsti.construction.list_lgst_gatestrings(
        myspecs, gsTarget.gates.keys())
    lsgstStrings = pygsti.construction.make_lsgst_lists(
            gsTarget.gates.keys(), rhoStrs, EStrs, mygerms, maxLs, fidPairList )

    print len(myspecs[0]), " rho specifiers"
    print len(myspecs[1]), " effect specifiers"
    print len(mygerms), " germs"
    print len(lgstStrings), " total LGST gate strings"
    print len(lsgstStrings[-1]), " LSGST strings before thinning"
    
    lsgstStringsToUse = lsgstStrings
    allRequiredStrs = pygsti.remove_duplicates(lgstStrings + lsgstStrings[-1])
     
    
    gs_dataGen = gsTarget.depolarize(gate_noise=0.1)
    dsFake = pygsti.construction.generate_fake_data(
        gs_dataGen, allRequiredStrs, nSamples, sampleError="multinomial",
        seed=seed)

    #Run LGST to get starting gate set
    gs_lgst = pygsti.do_lgst(dsFake, myspecs, gsTarget,
                             svdTruncateTo=gsTarget.dim, verbosity=3)
    gs_lgst_go = pygsti.optimize_gauge(gs_lgst,"target",
                                       targetGateset=gs_dataGen)
    
    #Run full iterative LSGST
    tStart = time.time()
    all_gs_lsgst = pygsti.do_iterative_mc2gst(
        dsFake, gs_lgst_go, lsgstStringsToUse,
        minProbClipForWeighting=minProbClipForWeighting,
        probClipInterval=(-1e5,1e5),
        verbosity=1, memLimit=3*(1024)**3, returnAll=True, 
        useFreqWeightedChiSq=useFreqWeightedChiSq, comm=comm)
    tEnd = time.time()
    print "Time = ",(tEnd-tStart)/3600.0,"hours"
    
    return all_gs_lsgst, gs_dataGen
    
    
def runOneQubit(comm=None):
    from pygsti.construction import std1Q_XYI as std

    maxLengths = [0,1,2,4,8,16,32,64,128,256,512,1024] #still need to define this manually
    specs = pygsti.construction.build_spam_specs(
        std.fiducials, prep_labels=std.gs_target.get_prep_labels(),
        effect_labels=std.gs_target.get_effect_labels())

    gsets, dsGen = runMC2GSTAnalysis(specs, std.germs, std.gs_target,
                                          1234, maxLengths, nSamples=1000,
                                          comm=comm)
    return gsets


# from inline tests
# dproduct return ret:
            #DEBUG!!!
            #if wrtFilter is not None:
            #    chk_prod = self.dproduct(gatestring, flat, wrtFilter=None)
            #    for ii,i in enumerate(wrtFilter):
            #        assert(_np.linalg.norm(chk_prod[i]-ret[ii]) < 1e-6)
            #    taken_chk_prod = chk_prod.take( wrtFilter, axis=0 )
            #    assert(_np.linalg.norm(taken_chk_prod-ret) < 1e-6)


#bulk_product
            #DEBUG!!! compare answer with no parallelization
            #chk_ret = self.bulk_product(evalTree, bScale, comm=None)
#     if bScale
                #DEBUG!!! compare answer with no parallelization
                #assert(_np.linalg.norm(chk_ret[0]-Gs) < 1e-6)
                #assert(_np.linalg.norm(chk_ret[1]-scaleVals) < 1e-6)
  # else:
                #DEBUG!!! compare answer with no parallelization
                #assert(_np.linalg.norm(chk_ret-Gs) < 1e-6)

# bulk_dproduct
            #DEBUG!!! compare answer with no parallelization
            #chk_ret = self.bulk_dproduct(evalTree, flat, bReturnProds,
            #                             bScale, memLimit, comm=None, wrtFilter=None)
# if not bReturnProds and not bScale
                #DEBUG!!! compare answer with no parallelization
                #assert(_np.linalg.norm(chk_ret-dGs) < 1e-6)
# else:
                #DEBUG!!! compare answer with no parallelization
                #if bScale:
                #    assert(not flat) # for testing.  We don't ever used
                #          #scaling & flat together (and it's hard to scale
                #          # in this case b/c we need to use repeat as above)
                #    nw = _np.newaxis
                #    scaled_dGs = rest_of_result[-1][:,nw,nw,nw] * dGs
                #    scaled_chk = chk_ret[-1][:,nw,nw,nw] * chk_ret[0]
                #    assert(_np.linalg.norm(scaled_chk-scaled_dGs)< 1e-6)
                #else:
                #    assert(_np.linalg.norm(chk_ret[0]-dGs) < 1e-6)
                #
                #if bReturnProds:
                #    if bScale:
                #        scaled_Gs = rest_of_result[-1][:,nw,nw] * \
                #            rest_of_result[0]
                #        scaled_chk = chk_ret[-1][:,nw,nw] * chk_ret[1]
                #        assert(_np.linalg.norm(scaled_chk-scaled_Gs)< 1e-6)
                #    else:
                #        assert(_np.linalg.norm(chk_ret[1]-rest_of_result[0])
                #               < 1e-6 )
                #
                #
                #if False and _np.linalg.norm(chk_ret[0]-dGs) >= 1e-6:
                #    #if bScale:
                #    #    print "SCALED"
                #    #    print chk_ret[-1]
                #
                #    rank = comm.Get_rank()
                #    if rank == 0:
                #        print "DEBUG: parallel mismatch"
                #        print "len(all_results) = ",len(all_results)
                #        print "diff = ",_np.linalg.norm(chk_ret[0]-dGs)
                #        for row in range(dGs.shape[0]):
                #            rowA = my_results[0][row,:].flatten()
                #            rowB = all_results[rank][0][row,:].flatten()
                #            rowC = dGs[row,:].flatten()
                #            chk_C = chk_ret[0][row,:].flatten()
                #
                #            def sp(ar):
                #                for i,x in enumerate(ar):
                #                    if abs(x) > 1e-4:
                #                        print i,":", x
                #            def spc(ar1,ar2):
                #                for i,x in enumerate(ar1):
                #                    if (abs(x) > 1e-4 or abs(ar2[i]) > 1e-4): # and abs(x-ar2[i]) > 1e-6:
                #                        print i,":", x, ar2[i], "(", (x-ar2[i]), ")", "[",x/ar2[i],"]"
                #
                #            assert( _np.linalg.norm(rowA-rowB) < 1e-6)
                #            assert( _np.linalg.norm(rowC[0:len(rowA)]-rowA) < 1e-6)
                #            #if _np.linalg.norm(rowA) > 1e-6:
                #            if _np.linalg.norm(rowC - chk_C) > 1e-6:
                #                print "SCALE for row%d = %g" % (row,rest_of_result[-1][row])
                #                print "CHKSCALE for row%d = %g" % (row,chk_ret[-1][row])
                #                print "row%d diff = " % row, _np.linalg.norm(rowC - chk_C)
                #                print "row%d (rank%d)A = " % (row,rank)
                #                sp(rowA)
                #                print "row%d (all vs check) = " % row
                #                spc(rowC, chk_C)
                #
                #                assert(False)
                #    assert(False)
# bulk_pr
        #DEBUG!!!
        #if comm is not None:
        #    chk_vp = self.bulk_pr(spamLabel, evalTree, clipTo, check, comm=None)
        #    assert(_np.linalg.norm(chk_vp-vp) < 1e-6)
# bulk_dpr:
        #DEBUG!!!
        #if comm is not None:
        #    chk = self.bulk_dpr(spamLabel, evalTree, 
        #                        returnPr,clipTo,check,memLimit,comm=None)
        #    if returnPr:
        #        assert(_np.linalg.norm(chk[0]-vdp) < 1e-6)
        #        assert(_np.linalg.norm(chk[1]-vp) < 1e-6)
        #    else:
        #        assert(_np.linalg.norm(chk-vdp) < 1e-6)





@mpitest(4)
def test_MPI(comm):
    #Individual processors
    my1ProcResults = runOneQubit(None)

    #Using all processors
    myManyProcResults = runOneQubit(comm)

    #compare on root proc
    if comm.Get_rank() == 0:
        for gs1,gs2 in zip(my1ProcResults,myManyProcResults):
            gs2_go = pygsti.optimize_gauge(gs2, "target", targetGateset=gs1,
                                           gateWeight=1.0, spamWeight=1.0)
            print "Frobenius distance = ", gs1.frobeniusdist(gs2_go)
            assert(gs1.frobeniusdist(gs2_go) < 1e-5)
    return



if __name__ == "__main__":
    unittest.main(verbosity=2)
    #oneQubitTest(g_comm)
    #oneQubitTest_Tutorial()
    #twoQubitTest()
    #test_MPI()

