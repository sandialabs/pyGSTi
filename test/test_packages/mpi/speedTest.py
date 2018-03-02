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


def runMC2GSTAnalysis(myspecs, mygerms, gsTarget, seed,
                      maxLs = [1,2,4,8],
                      nSamples=1000, useFreqWeightedChiSq=False,
                      minProbClipForWeighting=1e-4, fidPairList=None,
                      comm=None):
    rhoStrs, EStrs = pygsti.construction.get_spam_strs(myspecs)
    lgstStrings = pygsti.construction.list_lgst_gatestrings(
        myspecs, list(gsTarget.gates.keys()))
    lsgstStrings = pygsti.construction.make_lsgst_lists(
            list(gsTarget.gates.keys()), rhoStrs, EStrs, mygerms, maxLs, fidPairList )

    print(len(myspecs[0]), " rho specifiers")
    print(len(myspecs[1]), " effect specifiers")
    print(len(mygerms), " germs")
    print(len(lgstStrings), " total LGST gate strings")
    print(len(lsgstStrings[-1]), " LSGST strings before thinning")

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
    print("Time = ",(tEnd-tStart)/3600.0,"hours ( =",(tEnd-tStart)," secs)")

    return all_gs_lsgst, gs_dataGen


def runOneQubit(comm=None):
    from pygsti.construction import std1Q_XYI as std

    maxLengths = [1,2,4,8,16,32,64,128,256,512] #still need to define this manually
    specs = pygsti.construction.build_spam_specs(
        std.fiducials, prep_labels=std.gs_target.get_prep_labels(),
        effect_labels=std.gs_target.get_effect_labels())

    gsets, dsGen = runMC2GSTAnalysis(specs, std.germs, std.gs_target,
                                          1234, maxLengths, nSamples=1000,
                                          comm=comm)
    return gsets



def test_MPI(comm):
    #Individual processors
    #my1ProcResults = runOneQubit(None)

    #Using all processors
    myManyProcResults = runOneQubit(comm)

    #compare on root proc
    #if comm.Get_rank() == 0:
    #    for gs1,gs2 in zip(my1ProcResults,myManyProcResults):
    #        gs2_go = pygsti.optimize_gauge(gs2, "target", targetGateset=gs1,
    #                                       gateWeight=1.0, spamWeight=1.0)
    #        print "Frobenius distance = ", gs1.frobeniusdist(gs2_go)
    #        assert(gs1.frobeniusdist(gs2_go) < 1e-5)
    return



if __name__ == "__main__":
    #oneQubitTest(g_comm)
    #oneQubitTest_Tutorial()
    #twoQubitTest()
    test_MPI(g_comm)
