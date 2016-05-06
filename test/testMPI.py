import unittest
import time
import sys
import numpy as np
from mpinoseutils import *

import pygsti
from pygsti.construction import std1Q_XYI as std


def runOneQubit_Tutorial():
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

    #results.create_full_report_pdf(confidenceLevel=95,
    #    filename="tutorial_files/MyEvenEasierReport.pdf",verbosity=2)


def runMC2GSTAnalysis(myspecs, mygerms, gsTarget, seed,
                      maxLs = [1,2,4,8],
                      nSamples=1000, useFreqWeightedChiSq=False,
                      minProbClipForWeighting=1e-4, fidPairList=None,
                      comm=None, distributeMethod="gatestrings"):
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
        useFreqWeightedChiSq=useFreqWeightedChiSq, comm=comm,
        distributeMethod=distributeMethod)
    tEnd = time.time()
    print "Time = ",(tEnd-tStart)/3600.0,"hours"
    
    return all_gs_lsgst, gs_dataGen
    
    
def runOneQubit(comm=None, distributeMethod="gatestrings"):
    maxLengths = [0,1,2,4,8,16,32,64,128,256,512,1024] #still need to define this manually
    specs = pygsti.construction.build_spam_specs(
        std.fiducials, prep_labels=std.gs_target.get_prep_labels(),
        effect_labels=std.gs_target.get_effect_labels())

    gsets, dsGen = runMC2GSTAnalysis(specs, std.germs, std.gs_target,
                                          1234, maxLengths, nSamples=1000,
                                          comm=comm, distributeMethod=distributeMethod)
    return gsets


@mpitest(4)
def test_MPI_products(comm):

    #Create some gateset
    gs = std.gs_target.copy()
    gs.kick(0.1,seed=1234)

    #Get some gate strings
    maxLengths = [0,1,2,4,8,16,32,64,128,256]
    gstrs = pygsti.construction.make_lsgst_experiment_list(
        std.gs_target.gates.keys(), std.fiducials, std.fiducials, std.germs, maxLengths)
    tree = gs.bulk_evaltree(gstrs)

    #Check wrtFilter functionality in dproduct
    some_wrtFilter = [0,2,3,5,10]
    for s in gstrs[0:20]:
        result = gs._calc().dproduct(s, wrtFilter=some_wrtFilter)
        chk_result = gs.dproduct(s) #no filtering
        for ii,i in enumerate(some_wrtFilter):
            assert(np.linalg.norm(chk_result[i]-result[ii]) < 1e-6)
        taken_chk_result = chk_result.take( some_wrtFilter, axis=0 )
        assert(np.linalg.norm(taken_chk_result-result) < 1e-6)

    #Check products
    serial = gs.bulk_product(tree, bScale=False)
    parallel = gs.bulk_product(tree, bScale=False, comm=comm)
    assert(np.linalg.norm(serial-parallel) < 1e-6)

    serial, sscale = gs.bulk_product(tree, bScale=True)
    parallel, pscale = gs.bulk_product(tree, bScale=True, comm=comm)
    assert(np.linalg.norm(serial*sscale[:,None,None] - 
                          parallel*pscale[:,None,None]) < 1e-6)
    
    serial = gs.bulk_dproduct(tree, bScale=False)
    parallel = gs.bulk_dproduct(tree, bScale=False, comm=comm)
    assert(np.linalg.norm(serial-parallel) < 1e-6)

    serial, sscale = gs.bulk_dproduct(tree, bScale=True)
    parallel, pscale = gs.bulk_dproduct(tree, bScale=True, comm=comm)
    assert(np.linalg.norm(serial*sscale[:,None,None,None] - 
                          parallel*pscale[:,None,None,None]) < 1e-6)

    serial = gs.bulk_pr('plus', tree, clipTo=(-1e6,1e6))
    parallel = gs.bulk_pr('plus', tree, clipTo=(-1e6,1e6), comm=comm)
    assert(np.linalg.norm(serial-parallel) < 1e-6)

    serial = gs.bulk_dpr('plus', tree, clipTo=(-1e6,1e6))
    parallel = gs.bulk_dpr('plus', tree, clipTo=(-1e6,1e6), comm=comm)
    assert(np.linalg.norm(serial-parallel) < 1e-6)

    serial, sp = gs.bulk_dpr('plus', tree, returnPr=True, clipTo=(-1e6,1e6))
    parallel, pp = gs.bulk_dpr('plus', tree, returnPr=True, clipTo=(-1e6,1e6), comm=comm)
    assert(np.linalg.norm(serial-parallel) < 1e-6)
    assert(np.linalg.norm(sp-pp) < 1e-6)


#SCRATCH
#if np.linalg.norm(chk_ret[0]-dGs) >= 1e-6:
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





@mpitest(4)
def test_MPI_gatestrings(comm):
    #Individual processors
    my1ProcResults = runOneQubit()

    #Using all processors
    myManyProcResults = runOneQubit(comm,"gatestrings")

    #compare on root proc
    if comm.Get_rank() == 0:
        for gs1,gs2 in zip(my1ProcResults,myManyProcResults):
            gs2_go = pygsti.optimize_gauge(gs2, "target", targetGateset=gs1,
                                           gateWeight=1.0, spamWeight=1.0)
            print "Frobenius distance = ", gs1.frobeniusdist(gs2_go)
            assert(gs1.frobeniusdist(gs2_go) < 1e-5)
    return

@mpitest(4)
def test_MPI_derivcols(comm):
    #Individual processors
    my1ProcResults = runOneQubit()

    #Using all processors
    myManyProcResults = runOneQubit(comm,"deriv")

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
