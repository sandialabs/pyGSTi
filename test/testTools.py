import unittest
import numpy as np
import scipy.linalg

import pygsti
from pygsti.construction import std1Q_XYI as std


class ToolsTestCase(unittest.TestCase):

    def setUp(self):
        #Set GateSet objects to "strict" mode for testing
        pygsti.objects.GateSet._strict = True


    def assertArraysAlmostEqual(self,a,b):
        self.assertAlmostEqual( np.linalg.norm(a-b), 0 )


class ToolsMethods(ToolsTestCase):

    ###########################################################
    ## BASIS TOOLS TESTS     ##################################
    ###########################################################

    def test_expand_contract(self):
        # matrix that operates on 2x2 density matrices, but only on the 0-th and 3-rd
        # elements which correspond to the diagonals of the 2x2 density matrix.
        mxInStdBasis = np.array([[1,0,0,2],
                                 [0,0,0,0],
                                 [0,0,0,0],
                                 [3,0,0,4]],'d')

        # Reduce to a matrix operating on a density matrix space with 2 1x1 blocks (hence [1,1])
        mxInReducedBasis = pygsti.contract_to_std_direct_sum_mx(mxInStdBasis,[1,1])
        notReallyContracted = pygsti.contract_to_std_direct_sum_mx(mxInStdBasis,4)
        correctAnswer = np.array([[ 1.0,  2.0],
                                  [ 3.0,  4.0]])
        self.assertArraysAlmostEqual( mxInReducedBasis, correctAnswer )
        self.assertArraysAlmostEqual( notReallyContracted, mxInStdBasis )
        
        expandedMx = pygsti.expand_from_std_direct_sum_mx(mxInReducedBasis,[1,1])
        expandedMxAgain = pygsti.expand_from_std_direct_sum_mx(expandedMx,4)
        self.assertArraysAlmostEqual( expandedMx, mxInStdBasis )
        self.assertArraysAlmostEqual( expandedMxAgain, mxInStdBasis )

    def test_GellMann(self):

        id2x2 = np.array([[1,0],[0,1]])
        sigmax = np.array([[0,1],[1,0]])
        sigmay = np.array([[0,-1.0j],[1.0j,0]])
        sigmaz = np.array([[1,0],[0,-1]])

        # Gell-Mann 2x2 matrices should just be the sigma matrices
        GM2_mxs = pygsti.gm_matrices_unnormalized(2)
        self.assertTrue(len(GM2_mxs) == 4)
        self.assertArraysAlmostEqual( GM2_mxs[0], id2x2 )
        self.assertArraysAlmostEqual( GM2_mxs[1], sigmax )
        self.assertArraysAlmostEqual( GM2_mxs[2], sigmay )
        self.assertArraysAlmostEqual( GM2_mxs[3], sigmaz )
        with self.assertRaises(ValueError):
            pygsti.gm_matrices_unnormalized("FooBar") #arg must be tuple,list,or int

        # GM [1,1] matrices are the basis matrices for each block, concatenated together
        GM11_mxs = pygsti.gm_matrices_unnormalized([1,1]) 
        self.assertTrue(len(GM11_mxs) == 2)
        self.assertArraysAlmostEqual( GM11_mxs[0], np.array([[1,0],[0,0]],'d') )
        self.assertArraysAlmostEqual( GM11_mxs[1], np.array([[0,0],[0,1]],'d') )

        # Normalized Gell-Mann 2x2 matrices should just be the sigma matrices / sqrt(2)
        NGM2_mxs = pygsti.gm_matrices(2)
        self.assertTrue(len(NGM2_mxs) == 4)
        self.assertArraysAlmostEqual( NGM2_mxs[0], id2x2/np.sqrt(2) )
        self.assertArraysAlmostEqual( NGM2_mxs[1], sigmax/np.sqrt(2) )
        self.assertArraysAlmostEqual( NGM2_mxs[2], sigmay/np.sqrt(2) )
        self.assertArraysAlmostEqual( NGM2_mxs[3], sigmaz/np.sqrt(2) )
        
        #TODO: test 4x4 matrices?

    def test_orthogonality(self):

        #Gell Mann
        dim = 5
        mxs = pygsti.gm_matrices(dim)
        N = len(mxs); self.assertTrue(N == dim**2)

        gm_trMx = np.zeros((N,N), 'complex')
        for i in range(N):
            for j in range(N):
                gm_trMx[i,j] = np.trace(np.dot(np.conjugate(np.transpose(mxs[i])),mxs[j]))
                #Note: conjugate transpose not needed since mxs are Hermitian
        self.assertArraysAlmostEqual( gm_trMx, np.identity(N,'complex') )

        #Std Basis
        dim = 5
        mxs = pygsti.std_matrices(dim)
        N = len(mxs); self.assertTrue(N == dim**2)

        std_trMx = np.zeros((N,N), 'complex')
        for i in range(N):
            for j in range(N):
                std_trMx[i,j] = np.trace(np.dot(np.conjugate(np.transpose(mxs[i])),mxs[j]))
        self.assertArraysAlmostEqual( std_trMx, np.identity(N,'complex') )

        #Pauli-product basis
        dim = 4
        mxs = pygsti.pp_matrices(dim)
        N = len(mxs); self.assertTrue(N == dim**2)

        with self.assertRaises(ValueError):
            pygsti.pp_matrices("Foobar") #dim must be an int
        with self.assertRaises(ValueError):
            pygsti.pp_matrices(3) #dim must be a power of 2

        specialCase = pygsti.pp_matrices(1) #single 1x1 identity mx
        self.assertEqual( specialCase, [ np.identity(1,'complex') ] )

        pp_trMx = np.zeros((N,N), 'complex')
        for i in range(N):
            for j in range(N):
                pp_trMx[i,j] = np.trace(np.dot(np.conjugate(np.transpose(mxs[i])),mxs[j]))
                #Note: conjugate transpose not needed since mxs are Hermitian
        self.assertArraysAlmostEqual( pp_trMx, np.identity(N,'complex') )
        

    def test_transforms(self):
        mxStd = np.array([[1,0,0,0],
                          [0,1,0,0],
                          [0,0,1,0],
                          [0,0,0,1]], 'complex')
        vecStd = np.array([1,0,0,0], 'complex')

        mxGM = pygsti.std_to_gm(mxStd)
        mxStd2 = pygsti.gm_to_std(mxGM)
        self.assertArraysAlmostEqual( mxStd, mxStd2 )

        vecGM = pygsti.std_to_gm(vecStd)
        vecStd2 = pygsti.gm_to_std(vecGM)
        self.assertArraysAlmostEqual( vecStd, vecStd2 )
        
        mxPP = pygsti.std_to_pp(mxStd)
        mxStd2 = pygsti.pp_to_std(mxPP)
        self.assertArraysAlmostEqual( mxStd, mxStd2 )

        vecPP = pygsti.std_to_pp(vecStd)
        vecStd2 = pygsti.pp_to_std(vecPP)
        self.assertArraysAlmostEqual( vecStd, vecStd2 )

        mxPP2 = pygsti.gm_to_pp(mxGM)
        self.assertArraysAlmostEqual( mxPP, mxPP2 )

        vecPP2 = pygsti.gm_to_pp(vecGM)
        self.assertArraysAlmostEqual( vecPP, vecPP2 )
        
        mxGM2 = pygsti.pp_to_gm(mxPP)
        self.assertArraysAlmostEqual( mxGM, mxGM2 )

        vecGM2 = pygsti.pp_to_gm(vecPP)
        self.assertArraysAlmostEqual( vecGM, vecGM2 )

        
        non_herm_mxStd = np.array([[1,0,2,3j],
                                   [0,1,0,2],
                                   [0,0,1,0],
                                   [0,0,0,1]], 'complex')
        non_herm_vecStd = np.array([1,0,2,3j], 'complex') # ~ non-herm 2x2 density mx
        rank3tensor = np.ones((4,4,4),'d')

        with self.assertRaises(ValueError):
            pygsti.std_to_gm(non_herm_mxStd) #will result in gm mx with *imag* part
        with self.assertRaises(ValueError):
            pygsti.std_to_gm(non_herm_vecStd) #will result in gm vec with *imag* part
        with self.assertRaises(ValueError):
            pygsti.std_to_pp(non_herm_mxStd) #will result in pp mx with *imag* part
        with self.assertRaises(ValueError):
            pygsti.std_to_pp(non_herm_vecStd) #will result in pp vec with *imag* part

        with self.assertRaises(ValueError):
            pygsti.std_to_gm(rank3tensor) #only convert rank 1 & 2 objects
        with self.assertRaises(ValueError):
            pygsti.gm_to_std(rank3tensor) #only convert rank 1 & 2 objects
        with self.assertRaises(ValueError):
            pygsti.std_to_pp(rank3tensor) #only convert rank 1 & 2 objects
        with self.assertRaises(ValueError):
            pygsti.pp_to_std(rank3tensor) #only convert rank 1 & 2 objects
        with self.assertRaises(ValueError):
            pygsti.gm_to_pp(rank3tensor) #only convert rank 1 & 2 objects
        with self.assertRaises(ValueError):
            pygsti.pp_to_gm(rank3tensor) #only convert rank 1 & 2 objects

        densityMx = np.array( [[1,0],[0,-1]], 'complex' )
        gmVec = pygsti.stdmx_to_gmvec(densityMx)
        ppVec = pygsti.stdmx_to_ppvec(densityMx)
        stdVec = pygsti.stdmx_to_stdvec(densityMx)
        self.assertArraysAlmostEqual( gmVec, np.array( [[0],[0],[0],[np.sqrt(2)]], 'd') )
        self.assertArraysAlmostEqual( ppVec, np.array( [[0],[0],[0],[np.sqrt(2)]], 'd') )
        self.assertArraysAlmostEqual( stdVec, np.array( [[1],[0],[0],[-1]], 'complex') )

        mxFromGM  = pygsti.gmvec_to_stdmx(gmVec)
        mxFromPP  = pygsti.ppvec_to_stdmx(ppVec)
        mxFromStd = pygsti.stdvec_to_stdmx(stdVec)
        self.assertArraysAlmostEqual( mxFromGM, densityMx)
        self.assertArraysAlmostEqual( mxFromPP, densityMx)
        self.assertArraysAlmostEqual( mxFromStd, densityMx)


        

    def test_few_qubit_fns(self):
        state_vec = np.array([1,0],'complex')
        dmVec = pygsti.state_to_pauli_density_vec(state_vec)
        self.assertArraysAlmostEqual(dmVec, np.array([[0.70710678],[0],[0],[0.70710678]], 'complex'))

        theta = np.pi
        ex = 1j * theta*pygsti.sigmax/2
        U = scipy.linalg.expm(ex) 
        # U is 2x2 unitary matrix operating on single qubit in [0,1] basis (X(pi) rotation)

        op = pygsti.unitary_to_pauligate_1q(U)
        op_ans = np.array([[ 1.,  0.,  0.,  0.],
                           [ 0.,  1.,  0.,  0.],
                           [ 0.,  0., -1.,  0.],
                           [ 0.,  0.,  0., -1.]], 'd')
        self.assertArraysAlmostEqual(op, op_ans)

        U_2Q = np.identity(4, 'complex'); U_2Q[2:,2:] = U
        # U_2Q is 4x4 unitary matrix operating on isolated two-qubit space (CX(pi) rotation)
        
        op_2Q = pygsti.unitary_to_pauligate_2q(U_2Q)

        stdMx = np.array( [[1,0],[0,0]], 'complex' ) #density matrix
        pauliVec = pygsti.stdmx_to_ppvec(stdMx)
        self.assertArraysAlmostEqual(pauliVec, np.array([[0.70710678],[0],[0],[0.70710678]], 'complex'))

        stdMx2 = pygsti.ppvec_to_stdmx(pauliVec)
        self.assertArraysAlmostEqual( stdMx, stdMx2 )

    def test_basistools_misc(self):
        with self.assertRaises(ValueError):
            pygsti.tools.basistools._processBlockDims("FooBar") #arg should be a list,tuple,or int
        
            

    ###########################################################
    ## Chi2 and logL TESTS   ##################################
    ###########################################################
            
    def test_chi2_fn(self):
        ds = pygsti.objects.DataSet(fileToLoadFrom="cmp_chk_files/analysis.dataset")
        chi2, grad = pygsti.chi2(ds, std.gs_target, returnGradient=True)

        pygsti.gate_string_chi2( ('Gx',), ds, std.gs_target)
        pygsti.chi2fn_2outcome( N=100, p=0.5, f=0.6)
        pygsti.chi2fn_2outcome_wfreqs( N=100, p=0.5, f=0.6)
        pygsti.chi2fn( N=100, p=0.5, f=0.6)
        pygsti.chi2fn_wfreqs( N=100, p=0.5, f=0.6)

        with self.assertRaises(ValueError):
            pygsti.chi2(ds, std.gs_target, useFreqWeightedChiSq=True) #no impl yet

    def test_logl_fn(self):
        ds = pygsti.objects.DataSet(fileToLoadFrom="cmp_chk_files/analysis.dataset")
        gatestrings = pygsti.construction.gatestring_list( [ ('Gx',), ('Gy',), ('Gx','Gx') ] )
        spam_labels = std.gs_target.get_spam_labels()
        pygsti.create_count_vec_dict( spam_labels, ds, gatestrings )
        
        L1 = pygsti.logl(std.gs_target, ds, gatestrings, 
                         probClipInterval=(-1e6,1e6), countVecMx=None,
                         poissonPicture=True, check=False)
        L2 = pygsti.logl(std.gs_target, ds, gatestrings, 
                         probClipInterval=(-1e6,1e6), countVecMx=None,
                         poissonPicture=False, check=False) #Non-poisson-picture

        dL1 = pygsti.logl_jacobian(std.gs_target, ds, gatestrings,
                                   probClipInterval=(-1e6,1e6), radius=1e-4, 
                                   poissonPicture=True, check=False)
        dL2 = pygsti.logl_jacobian(std.gs_target, ds, gatestrings,
                                   probClipInterval=(-1e6,1e6), radius=1e-4, 
                                   poissonPicture=False, check=False)
        dL2b = pygsti.logl_jacobian(std.gs_target, ds, None,
                                   probClipInterval=(-1e6,1e6), radius=1e-4, 
                                   poissonPicture=False, check=False) #test None as gs list


        hL1 = pygsti.logl_hessian(std.gs_target, ds, gatestrings,
                                  probClipInterval=(-1e6,1e6), radius=1e-4, 
                                  poissonPicture=True, check=False)

        hL2 = pygsti.logl_hessian(std.gs_target, ds, gatestrings,
                                  probClipInterval=(-1e6,1e6), radius=1e-4, 
                                  poissonPicture=False, check=False)
        hL2b = pygsti.logl_hessian(std.gs_target, ds, None,
                                   probClipInterval=(-1e6,1e6), radius=1e-4, 
                                   poissonPicture=False, check=False) #test None as gs list


        maxL1 = pygsti.logl_max(ds, gatestrings, poissonPicture=True, check=True)
        maxL2 = pygsti.logl_max(ds, gatestrings, poissonPicture=False, check=True)

        pygsti.cptp_penalty(std.gs_target, include_spam_penalty=True)
        twoDelta1 = pygsti.two_delta_loglfn(N=100, p=0.5, f=0.6, minProbClip=1e-6, poissonPicture=True)
        twoDelta2 = pygsti.two_delta_loglfn(N=100, p=0.5, f=0.6, minProbClip=1e-6, poissonPicture=False)
        

    def test_gate_tools(self):
        oneRealPair = np.array( [[1+1j, 0, 0, 0],
                             [ 0, 1-1j,0, 0],
                             [ 0,   0, 2, 0],
                             [ 0,   0,  0, 2]], 'complex')
        decomp = pygsti.decompose_gate_matrix(oneRealPair) 
            #decompose gate mx whose eigenvalues have a real but non-unit pair

        dblRealPair = np.array( [[3, 0, 0, 0],
                             [ 0, 3,0, 0],
                             [ 0,   0, 2, 0],
                             [ 0,   0,  0, 2]], 'complex')
        decomp = pygsti.decompose_gate_matrix(dblRealPair) 
            #decompose gate mx whose eigenvalues have two real but non-unit pairs


        unpairedMx = np.array( [[1+1j, 0, 0, 0],
                                [ 0, 2-1j,0, 0],
                                [ 0,   0, 2+2j, 0],
                                [ 0,   0,  0,  1.0+3j]], 'complex')
        decomp = pygsti.decompose_gate_matrix(unpairedMx) 
            #decompose gate mx which has all complex eigenvalue -> bail out
        self.assertFalse(decomp['isValid'])

        largeMx = np.identity(16,'d')
        decomp = pygsti.decompose_gate_matrix(largeMx) #can only handle 1Q mxs
        self.assertFalse(decomp['isValid'])

        A = np.array( [[0.9, 0, 0.1j, 0],
                       [ 0,  0, 0,    0],
                       [ -0.1j, 0, 0, 0],
                       [ 0,  0,  0,  0.1]], 'complex')

        B = np.array( [[0.5, 0, 0, -0.2j],
                       [ 0,  0.25, 0,  0],
                       [ 0, 0, 0.25,   0],
                       [ 0.2j,  0,  0,  0.1]], 'complex')

        self.assertAlmostEqual( pygsti.frobeniusdist(A,A), 0.0 )
        self.assertAlmostEqual( pygsti.jtracedist(A,A,mxBasis="std"), 0.0 )
        self.assertAlmostEqual( pygsti.diamonddist(A,A,mxBasis="std"), 0.0 )
        self.assertAlmostEqual( pygsti.frobeniusdist(A,B), (0.430116263352+0j) )
        self.assertAlmostEqual( pygsti.jtracedist(A,B,mxBasis="std"), 0.260078105936)
        self.assertAlmostEqual( pygsti.diamonddist(A,B,mxBasis="std"), 0.614258836298)

        self.assertAlmostEqual( pygsti.frobeniusdist(A,B), pygsti.frobeniusnorm(A-B) )
        self.assertAlmostEqual( pygsti.frobeniusdist(A,B), np.sqrt( pygsti.frobeniusnorm2(A-B) ) )
        

    def test_jamiolkowski_ops(self):
        mxGM  = np.array([[1, 0, 0, 0],
                          [0, 0, 1, 0],
                          [0,-1, 0, 0],
                          [0, 0, 0, 1]], 'complex')

        mxStd = pygsti.gm_to_std(mxGM)
        mxPP  = pygsti.gm_to_pp(mxGM)

        choiStd = pygsti.jamiolkowski_iso(mxStd, "std","std")
        choiStd2 = pygsti.jamiolkowski_iso(mxGM, "gm","std")
        choiStd3 = pygsti.jamiolkowski_iso(mxPP, "pp","std")

        choiGM = pygsti.jamiolkowski_iso(mxStd, "std","gm")
        choiGM2 = pygsti.jamiolkowski_iso(mxGM, "gm","gm")
        choiGM3 = pygsti.jamiolkowski_iso(mxPP, "pp","gm")

        choiPP = pygsti.jamiolkowski_iso(mxStd, "std","pp")
        choiPP2 = pygsti.jamiolkowski_iso(mxGM, "gm","pp")
        choiPP3 = pygsti.jamiolkowski_iso(mxPP, "pp","pp")

        self.assertArraysAlmostEqual( choiStd, choiStd2)
        self.assertArraysAlmostEqual( choiStd, choiStd3)
        self.assertArraysAlmostEqual( choiGM, choiGM2)
        self.assertArraysAlmostEqual( choiGM, choiGM3)
        self.assertArraysAlmostEqual( choiPP, choiPP2)
        self.assertArraysAlmostEqual( choiPP, choiPP3)

        gateStd = pygsti.jamiolkowski_iso_inv(choiStd, "std","std")
        gateStd2 = pygsti.jamiolkowski_iso_inv(choiGM, "gm","std")
        gateStd3 = pygsti.jamiolkowski_iso_inv(choiPP, "pp","std")

        gateGM = pygsti.jamiolkowski_iso_inv(choiStd, "std","gm")
        gateGM2 = pygsti.jamiolkowski_iso_inv(choiGM, "gm","gm")
        gateGM3 = pygsti.jamiolkowski_iso_inv(choiPP, "pp","gm")

        gatePP = pygsti.jamiolkowski_iso_inv(choiStd, "std","pp")
        gatePP2 = pygsti.jamiolkowski_iso_inv(choiGM, "gm","pp")
        gatePP3 = pygsti.jamiolkowski_iso_inv(choiPP, "pp","pp")

        self.assertArraysAlmostEqual( gateStd, mxStd)
        self.assertArraysAlmostEqual( gateStd2, mxStd)
        self.assertArraysAlmostEqual( gateStd3, mxStd)

        self.assertArraysAlmostEqual( gateGM,  mxGM)
        self.assertArraysAlmostEqual( gateGM2, mxGM)
        self.assertArraysAlmostEqual( gateGM3, mxGM)

        self.assertArraysAlmostEqual( gatePP,  mxPP)
        self.assertArraysAlmostEqual( gatePP2, mxPP)
        self.assertArraysAlmostEqual( gatePP3, mxPP)


        with self.assertRaises(ValueError):
            pygsti.jamiolkowski_iso(mxStd, "foobar","gm") #invalid gate basis
        with self.assertRaises(ValueError):
            pygsti.jamiolkowski_iso(mxStd, "std","foobar") #invalid choi basis
        with self.assertRaises(ValueError):
            pygsti.jamiolkowski_iso_inv(choiStd, "foobar","gm") #invalid choi basis
        with self.assertRaises(ValueError):
            pygsti.jamiolkowski_iso_inv(choiStd, "std","foobar") #invalid gate basis
        

        sumOfNeg  = pygsti.sum_of_negative_choi_evals(std.gs_target)
        sumsOfNeg = pygsti.sums_of_negative_choi_evals(std.gs_target)
        magsOfNeg = pygsti.mags_of_negative_choi_evals(std.gs_target)
        self.assertAlmostEqual(sumOfNeg, 0.0)
        self.assertArraysAlmostEqual(sumsOfNeg, np.zeros(3,'d')) # 3 gates in std.gs_target
        self.assertArraysAlmostEqual(magsOfNeg, np.zeros(12,'d')) # 3 gates * 4 evals each = 12

    def test_matrixtools(self):
        herm_mx = np.array( [[ 1, 1+2j],
                             [1-2j, 3]], 'complex' )
        non_herm_mx = np.array( [[ 1, 4+2j],
                                 [1+2j, 3]], 'complex' )
        self.assertTrue( pygsti.is_hermitian(herm_mx) )
        self.assertFalse( pygsti.is_hermitian(non_herm_mx) )

        pos_mx = np.array( [[ 4, 0.2],
                             [0.1, 3]], 'complex' )
        non_pos_mx = np.array( [[ 0, 1],
                                 [1, 0]], 'complex' )
        self.assertTrue( pygsti.is_pos_def(pos_mx) )
        self.assertFalse( pygsti.is_pos_def(non_pos_mx) )

        density_mx = np.array( [[ 0.9,   0],
                                [   0, 0.1]], 'complex' )
        non_density_mx = np.array( [[ 2.0, 1.0],
                                    [-1.0,   0]], 'complex' )
        self.assertTrue( pygsti.is_valid_density_mx(density_mx) )
        self.assertFalse( pygsti.is_valid_density_mx(non_density_mx) )

        s1 = pygsti.mx_to_string(density_mx)
        s2 = pygsti.mx_to_string(non_herm_mx)
        #print "\n>%s<" % s1
        #print "\n>%s<" % s2
        
        #TODO: go through matrixtools.py and add tests for every function

    def test_rb_tools(self):
        ds = pygsti.objects.DataSet(fileToLoadFrom="cmp_chk_files/analysis.dataset")
        val = pygsti.rb_decay(0.1,0.1,0.1)
        self.assertAlmostEqual(val, 0.1039800665)
        decay = pygsti.rb_decay_rate(ds,showPlot=False,xlim=(0,10),ylim=(0,10),
                                     saveFigPath="temp_test_files/RBdecay.png")

    def test_rpe_tools(self):
        from pygsti.tools import rpe

        xhat = 10 #plus counts for sin string
        yhat = 90 #plus counts for cos string
        k = 1 #experiment generation
        Nx = 100 # sin string clicks
        Ny = 100 # cos string clicks
        k1Alpha = rpe.extract_rotation_hat(xhat,yhat,k,Nx,Ny,"alpha",
                                           previousAngle=None)
        k1Eps = rpe.extract_rotation_hat(xhat,yhat,k,Nx,Ny,"epsilon",
                                         previousAngle=None)
        self.assertAlmostEqual(k1Alpha, 0.785398163397)
        self.assertAlmostEqual(k1Eps, -2.35619449019)

        k = 2 #experiment generation
        k2Alpha = rpe.extract_rotation_hat(xhat,yhat,k,Nx,Ny,"alpha",
                                           previousAngle=k1Alpha)
        k2Eps = rpe.extract_rotation_hat(xhat,yhat,k,Nx,Ny,"epsilon",
                                         previousAngle=k1Eps)
        self.assertAlmostEqual(k2Alpha, 0.392699081699)
        self.assertAlmostEqual(k2Eps, -1.1780972451)


        with self.assertRaises(Exception):
            rpe.extract_rotation_hat(xhat,yhat,2,Nx,Ny,"epsilon",
                                     previousAngle=None) #need previous angle

        with self.assertRaises(Exception):
            rpe.extract_rotation_hat(xhat,yhat,1,Nx,Ny,"foobar") #bad angle name
        

        from pygsti.construction import std1Q_XZ as stdXZ
        target = stdXZ.gs_target.copy()
        target.gates['Gi'] =  std.gs_target.gates['Gi'] #need a Gi gate...
        stringListD = pygsti.construction.make_rpe_string_list_d(2)
        gs_depolXZ = target.depolarize(gate_noise=0.1,spam_noise=0.1)
        ds = pygsti.construction.generate_fake_data(gs_depolXZ, stringListD['totalStrList'],
                                                    nSamples=1000, sampleError='binomial')

        epslist = rpe.est_angle_list(ds,stringListD['epsilon','sin'],stringListD['epsilon','cos'],
                                     angleName="epsilon")

        tlist,dummy = rpe.est_theta_list(ds,stringListD['theta','sin'],stringListD['theta','cos'],
                                         epslist,returnPhiFunList=True)
        tlist = rpe.est_theta_list(ds,stringListD['theta','sin'],stringListD['theta','cos'],
                                   epslist,returnPhiFunList=False)

        alpha = rpe.extract_alpha( stdXZ.gs_target )
        eps = rpe.extract_epsilon( stdXZ.gs_target )
        theta = rpe.extract_theta( stdXZ.gs_target )
        rpe.analyze_simulated_rpe_experiment(ds,gs_depolXZ,stringListD)



      
if __name__ == "__main__":
    unittest.main(verbosity=2)
