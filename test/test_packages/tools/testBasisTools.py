from .toolsBaseCase import ToolsTestCase
import unittest
import numpy as np
import scipy
import pygsti

class BasisToolsTestCase(ToolsTestCase):

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

    def test_basis_longname(self):
        longnames = {pygsti.tools.basistools.basis_longname(basis) for basis in {'gm', 'std', 'pp'}}
        self.assertEqual(longnames, {'Gell-Mann', 'Matrix-unit', 'Pauli-prod'})
        self.assertEqual(pygsti.tools.basistools.basis_longname('not a basis'), '?Unknown?')

if __name__ == '__main__':
    unittest.main(verbosity=2)
