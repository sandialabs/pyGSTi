from ..testutils import BaseTestCase, compare_files, temp_files
import unittest
import numpy as np
import scipy
import pygsti

import pygsti.tools.basistools    as bt
import pygsti.tools.lindbladtools as lindbladtools

from pygsti.baseobjs import Basis, Dim

from functools import partial

class BasisBaseTestCase(BaseTestCase):

    def test_expand_contract(self):
        # matrix that operates on 2x2 density matrices, but only on the 0-th and 3-rd
        # elements which correspond to the diagonals of the 2x2 density matrix.
        mxInStdBasis = np.array([[1,0,0,2],
                                 [0,0,0,0],
                                 [0,0,0,0],
                                 [3,0,0,4]],'d')

        # Reduce to a matrix operating on a density matrix space with 2 1x1 blocks (hence [1,1])
        begin = Basis('std', [1,1])
        end   = Basis('std', 2)
        
        mxInReducedBasis = bt.resize_std_mx(mxInStdBasis, 'contract', end, begin)
        #mxInReducedBasis = bt.change_basis(mxInStdBasis, begin, end)
        notReallyContracted = bt.change_basis(mxInStdBasis, 'std', 'std', 4)
        correctAnswer = np.array([[ 1.0,  2.0],
                                  [ 3.0,  4.0]])
        #self.assertArraysAlmostEqual( mxInReducedBasis, correctAnswer )
        self.assertArraysAlmostEqual( notReallyContracted, mxInStdBasis )

        expandedMx = bt.resize_std_mx(mxInReducedBasis, 'expand', begin, end)
        #expandedMx = bt.change_basis(mxInReducedBasis, end, begin)
        expandedMxAgain = bt.change_basis(expandedMx, 'std', 'std', 4)
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

        '''
        # GM [1,1] matrices are the basis matrices for each block, concatenated together
        GM11_mxs = pygsti.gm_matrices_unnormalized([1,1])
        self.assertTrue(len(GM11_mxs) == 2)
        self.assertArraysAlmostEqual( GM11_mxs[0], np.array([[1,0],[0,0]],'d') )
        self.assertArraysAlmostEqual( GM11_mxs[1], np.array([[0,0],[0,1]],'d') )
        '''

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

        change = bt.change_basis
        mxGM = change(mxStd, 'std', 'gm')
        mxStd2 = change(mxGM, 'gm', 'std')
        self.assertArraysAlmostEqual( mxStd, mxStd2)

        vecGM = change(vecStd, 'std', 'gm')
        vecStd2 = change(vecGM, 'gm', 'std')
        self.assertArraysAlmostEqual( vecStd, vecStd2 )

        mxPP = change(mxStd, 'std', 'pp')
        mxStd2 = change(mxPP, 'pp', 'std')
        self.assertArraysAlmostEqual( mxStd, mxStd2 )

        vecPP = change(vecStd, 'std', 'pp')
        vecStd2 = change(vecPP, 'pp', 'std')
        self.assertArraysAlmostEqual( vecStd, vecStd2 )

        mxPP2 = change(mxGM, 'gm', 'pp')
        self.assertArraysAlmostEqual( mxPP, mxPP2 )

        vecPP2 = change(vecGM, 'gm', 'pp')
        self.assertArraysAlmostEqual( vecPP, vecPP2 )

        mxGM2 = change(mxPP, 'pp', 'gm')
        self.assertArraysAlmostEqual( mxGM, mxGM2 )

        vecGM2 = change(vecPP, 'pp', 'gm')
        self.assertArraysAlmostEqual( vecGM, vecGM2 )


        non_herm_mxStd = np.array([[1,0,2,3j],
                                   [0,1,0,2],
                                   [0,0,1,0],
                                   [0,0,0,1]], 'complex')
        non_herm_vecStd = np.array([1,0,2,3j], 'complex') # ~ non-herm 2x2 density mx
        rank3tensor = np.ones((4,4,4),'d')

        with self.assertRaises(ValueError):
            change(non_herm_mxStd, 'std', 'gm') #will result in gm mx with *imag* part
        with self.assertRaises(ValueError):
            change(non_herm_vecStd, 'std', 'gm') #will result in gm vec with *imag* part
        with self.assertRaises(ValueError):
            change(non_herm_mxStd, 'std', 'pp') #will result in pp mx with *imag* part
        with self.assertRaises(ValueError):
            change(non_herm_vecStd, 'std', 'pp') #will result in pp vec with *imag* part

        with self.assertRaises(ValueError):
            change(rank3tensor, 'std', 'gm') #only convert rank 1 & 2 objects
        with self.assertRaises(ValueError):
            change(rank3tensor, 'gm', 'std') #only convert rank 1 & 2 objects
        with self.assertRaises(ValueError):
            change(rank3tensor, 'std', 'pp') #only convert rank 1 & 2 objects
        with self.assertRaises(ValueError):
            change(rank3tensor, 'pp', 'std') #only convert rank 1 & 2 objects
        with self.assertRaises(ValueError):
            change(rank3tensor, 'gm', 'pp') #only convert rank 1 & 2 objects
        with self.assertRaises(ValueError):
            change(rank3tensor, 'pp', 'gm') #only convert rank 1 & 2 objects

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

        op = pygsti.unitary_to_pauligate(U)
        op_ans = np.array([[ 1.,  0.,  0.,  0.],
                           [ 0.,  1.,  0.,  0.],
                           [ 0.,  0., -1.,  0.],
                           [ 0.,  0.,  0., -1.]], 'd')
        self.assertArraysAlmostEqual(op, op_ans)

        U_2Q = np.identity(4, 'complex'); U_2Q[2:,2:] = U
        # U_2Q is 4x4 unitary matrix operating on isolated two-qubit space (CX(pi) rotation)

        op_2Q = pygsti.unitary_to_pauligate(U_2Q)

        stdMx = np.array( [[1,0],[0,0]], 'complex' ) #density matrix
        pauliVec = pygsti.stdmx_to_ppvec(stdMx)
        self.assertArraysAlmostEqual(pauliVec, np.array([[0.70710678],[0],[0],[0.70710678]], 'complex'))

        stdMx2 = pygsti.ppvec_to_stdmx(pauliVec)
        self.assertArraysAlmostEqual( stdMx, stdMx2 )

    def test_basis_misc(self):
        with self.assertRaises(TypeError):
            Dim("FooBar") #arg should be a list,tuple,or int
        bt.pp_matrices([1])

    def test_basis_longname(self):
        longnames = {bt.basis_longname(b) for b in {'gm', 'std', 'pp', 'qt'}}
        self.assertEqual(longnames, {'Gell-Mann', 'Matrix-unit', 'Pauli-Product', 'Qutrit'})
        with self.assertRaises(KeyError):
            bt.basis_longname('not a basis')

    def test_basis_element_labels(self):
        basisnames = ['gm', 'std', 'pp']

        # One dimensional gm
        self.assertEqual([''], bt.basis_element_labels('gm', 1))

        # Two dimensional
        expectedLabels = [
        ['I', 'X', 'Y', 'Z'],
        ['(0,0)', '(0,1)', '(1,0)', '(1,1)'],
        ['I', 'X', 'Y', 'Z']]
        labels = [bt.basis_element_labels(basisname, 2)  for basisname in basisnames]
        self.assertEqual(labels, expectedLabels)

        with self.assertRaises(NotImplementedError):
            bt.basis_element_labels('asdklfasdf', 2)

        # Non power of two for pp labels:
        with self.assertRaises(ValueError):
            label = bt.basis_element_labels('pp', 3)

        # Single list arg for pp labels
        self.assertEqual(bt.basis_element_labels('pp', [2]), ['I', 'X', 'Y', 'Z'])

        # Four dimensional+
        expectedLabels = [['I^{(0)}', 'X^{(0)}_{0,1}', 'X^{(0)}_{0,2}', 'X^{(0)}_{0,3}', 'X^{(0)}_{1,2}', 'X^{(0)}_{1,3}', 'X^{(0)}_{2,3}', 'Y^{(0)}_{0,1}', 'Y^{(0)}_{0,2}', 'Y^{(0)}_{0,3}', 'Y^{(0)}_{1,2}', 'Y^{(0)}_{1,3}', 'Y^{(0)}_{2,3}', 'Z^{(0)}_{1}', 'Z^{(0)}_{2}', 'Z^{(0)}_{3}'], ['(0,0)', '(0,1)', '(0,2)', '(0,3)', '(1,0)', '(1,1)', '(1,2)', '(1,3)', '(2,0)', '(2,1)', '(2,2)', '(2,3)', '(3,0)', '(3,1)', '(3,2)', '(3,3)'], ['II', 'IX', 'IY', 'IZ', 'XI', 'XX', 'XY', 'XZ', 'YI', 'YX', 'YY', 'YZ', 'ZI', 'ZX', 'ZY', 'ZZ']]
        labels = [bt.basis_element_labels(basisname, 4)  for basisname in basisnames]
        self.assertEqual(expectedLabels, labels)

    def test_hamiltonian_to_lindbladian(self):
        expectedLindbladian = np.array([[ 0,  0,  0,  0],
                                        [ 0,  0,  0,  0,],
                                        [ 0,  0,  0,  0,],
                                        [ 0,  0,  0,  0]]
                                       )

        self.assertArraysAlmostEqual(lindbladtools.hamiltonian_to_lindbladian(np.zeros(shape=(2,2))),
                                     expectedLindbladian)

    def test_vec_to_stdmx(self):
        vec = np.zeros(shape=(2,))
        for b in {'gm', 'pp', 'std'}:
            bt.vec_to_stdmx(vec, b)
        with self.assertRaises(NotImplementedError):
            bt.vec_to_stdmx(vec, 'akdfj;ladskf')

    def test_composite_basis(self):
        comp = Basis([('std', 2,), ('std', 1)])

        a = Basis([('std', 2), ('std', 2)])
        b = Basis('std', [2,2])
        self.assertEqual(len(a), len(b))
        self.assertArraysAlmostEqual(np.array(a._matrices), np.array(b._matrices))

    def test_auto_expand(self):
        comp = Basis(matrices=[('std', 2,), ('std', 1)])
        std  = Basis('std', 3)
        mxStd = np.identity(5)
        test   = bt.resize_std_mx(mxStd, 'expand', comp, std)
        test2  = bt.resize_std_mx(test, 'contract', std, comp)
        self.assertArraysAlmostEqual(test2, mxStd)

    def test_flexible_change_basis(self):
        comp  = Basis(matrices=[('gm', 2,), ('gm', 1)])
        std   = Basis('std', 3)
        mx    = np.identity(5)
        test  = bt.flexible_change_basis(mx, comp, std)
        self.assertEqual(test.shape[0], sum(comp.dim.blockDims) ** 2)
        test2 = bt.flexible_change_basis(test, std, comp)
        self.assertArraysAlmostEqual(test2, mx)

    def test_change_between_composites(self):
        a = Basis('std', [2, 1])
        b = Basis('gm',  [2, 1])
        mxStd = np.identity(5)
        test = bt.change_basis(mxStd, a, b)
        self.assertEqual(test.shape, mxStd.shape)
        test2 = bt.change_basis(test, b, a)
        self.assertArraysAlmostEqual(test2, mxStd)

    def test_qt(self):
        qt = Basis('qt', 3)
        qt = Basis('qt', [3])

    def test_general(self):
        Basis('pp', 2)
        Basis('std', [2, 1])
        Basis([('std', 2), ('gm', 2)])

        std  = Basis('std', 2)
        std4  = Basis('std', 4)
        std2x2 = Basis([('std', 2), ('std', 2)])
        gm   = Basis('gm', 2)
        ungm = Basis('gm_unnormalized', 2)
        empty = Basis([]) #special "empty" basis
        self.assertEqual(empty.name, "*Empty*")

        from_basis,to_basis = pygsti.tools.build_basis_pair(np.identity(4,'d'),"std","gm")
        from_basis,to_basis = pygsti.tools.build_basis_pair(np.identity(4,'d'),std,"gm")
        from_basis,to_basis = pygsti.tools.build_basis_pair(np.identity(4,'d'),"std",gm)

        gm_mxs = gm.get_composite_matrices()
        unnorm = Basis(matrices=[ gm_mxs[0], 2*gm_mxs[1] ])

        std[0]
        std.get_sub_basis_matrices(0)

        print(gm.get_composite_matrices())
        self.assertTrue(gm.is_normalized())
        self.assertFalse(ungm.is_normalized())
        self.assertFalse(unnorm.is_normalized())

        transMx = bt.transform_matrix(std, gm)

        composite = Basis([std, gm])

        comp = Basis(matrices=[std, gm], name='comp', longname='CustomComposite')

        comp.labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        comp = Basis(matrices=[std, gm], name='comp', longname='CustomComposite', labels=[
                'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'
            ])

        std2x2Matrices = np.array([
            [[1, 0],
             [0, 0]],
            
            [[0, 1],
             [0, 0]],
            
            [[0, 0],
             [1, 0]],
            
            [[0, 0],
             [0, 1]]
        ],'complex')

        empty = Basis(matrices=[])
        alt_standard = Basis(matrices=std2x2Matrices)
        print("MXS = \n",alt_standard._matrices)
        alt_standard = Basis(matrices=std2x2Matrices,
                             name='std',
                             longname='Standard'
                            )
        self.assertEqual(alt_standard, std2x2Matrices)

        mx = np.array([
                [1, 0, 0, 1],
                [0, 1, 2, 0],
                [0, 2, 1, 0],
                [1, 0, 0, 1]
            ])

        bt.change_basis(mx, 'std', 'gm') # shortname lookup
        bt.change_basis(mx, std, gm) # object
        bt.change_basis(mx, std, 'gm') # combination
        bt.flexible_change_basis(mx, std, gm) #same dimension
        I2x2 = np.identity(8,'d')
        I4 = bt.flexible_change_basis(I2x2, std2x2, std4)
        self.assertArraysAlmostEqual(bt.flexible_change_basis(I4, std4, std2x2), I2x2)
        
        with self.assertRaises(ValueError):
            bt.change_basis(mx, std, std4) # basis size mismatch
        
        
        mxInStdBasis = np.array([[1,0,0,2],
                                 [0,0,0,0],
                                 [0,0,0,0],
                                 [3,0,0,4]],'d')

        begin = Basis('std', [1,1])
        end   = Basis('std', 2)
        mxInReducedBasis = bt.resize_std_mx(mxInStdBasis, 'contract', end, begin)
        original         = bt.resize_std_mx(mxInReducedBasis, 'expand', begin, end)

    def test_basis_object(self):
        #test a few aspects of a Basis object that other tests miss...
        b = Basis("pp",2)
        beq = b.expanded_equivalent()
        longnm = bt.basis_longname(b)
        lbls = bt.basis_element_labels(b)

        raw_mxs = bt.basis_matrices("pp",2)
        with self.assertRaises(NotImplementedError):
            bt.basis_matrices("foobar",2) #invalid basis name

        print("Dim = ", repr(b.dim) ) # calls Dim.__repr__

    def test_basis_constructors(self):
        #special constructions not covered elsewhere
        ppMax1 = bt.pp_matrices(2,maxWeight=1) #using maxWeight
        qutrit1 = bt.qt_matrices(1) #special case when dim==1

        #Cover invalid Dim construction
        with self.assertRaises(TypeError):
            pygsti.baseobjs.Dim(1.2)

        
    def test_sparse_basis(self):
        sparsePP = Basis("pp",2,sparse=True)
        sparsePP2 = Basis("pp",2,sparse=True)
        sparseBlockPP = Basis("pp",[2,2],sparse=True)
        sparsePP_2Q = Basis("pp",4,sparse=True)
        sparseGM_2Q = Basis("gm",2,sparse=True) #different sparsity structure than PP 2Q
        denseGM = Basis("gm",2,sparse=False)
        
        mxs = sparsePP.get_composite_matrices()
        block_mxs = sparseBlockPP.get_composite_matrices()

        expeq = sparsePP.expanded_equivalent()
        block_expeq = sparseBlockPP.expanded_equivalent()

        raw_mxs = bt.basis_matrices("pp",2,sparse=True)
        
        #test equality of bases with other bases and matrices
        self.assertEqual(sparsePP, sparsePP2)
        self.assertEqual(sparsePP, raw_mxs)
        self.assertNotEqual(sparsePP, sparsePP_2Q)
        self.assertNotEqual(sparsePP_2Q, sparseGM_2Q)

        #sparse transform matrix
        trans = sparsePP.transform_matrix(sparsePP2)
        self.assertArraysAlmostEqual(trans, np.identity(4,'d'))
        trans2 = sparsePP.transform_matrix(denseGM)

        #test equality for large bases, which is too expensive so it always returns false
        large_sparsePP = Basis("pp",16,sparse=True)
        large_sparsePP2 = Basis("pp",16,sparse=True)
        self.assertNotEqual(large_sparsePP, large_sparsePP2)
        

        
        


if __name__ == '__main__':
    unittest.main(verbosity=2)
