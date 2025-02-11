import numpy as np
import scipy.sparse as sps

from pygsti.tools import lindbladtools as lt
from pygsti.modelmembers.operations import LindbladErrorgen
from pygsti.baseobjs import Basis, QubitSpace
from pygsti.baseobjs.errorgenlabel import GlobalElementaryErrorgenLabel, LocalElementaryErrorgenLabel
from ..util import BaseCase


class LindbladToolsTester(BaseCase):
    def test_hamiltonian_to_lindbladian(self):
        expectedLindbladian = np.array([
            [ 0,  0,  0,  0],
            [ 0,  0,  0,  0],
            [ 0,  0,  0,  0],
            [ 0,  0,  0,  0]
        ])
        self.assertArraysAlmostEqual(lt.create_elementary_errorgen('H', np.zeros(shape=(2, 2))),
                                     expectedLindbladian)
        sparse = sps.csr_matrix(np.zeros(shape=(2, 2)))
        #spL = lt.hamiltonian_to_lindbladian(sparse, True)
        spL = lt.create_elementary_errorgen('H', sparse, sparse=True)
        self.assertArraysAlmostEqual(spL.toarray(),
                                     expectedLindbladian)

    def test_stochastic_lindbladian(self):
        a = np.array([[1, 2], [3, 4]], 'd')
        expected = np.array([
            [ 1,  2,  2,  4],
            [ 3,  4,  6,  8],
            [ 3,  6,  4,  8],
            [ 9, 12, 12, 16]
        ], 'd')
        dual_eg, norm = lt.create_elementary_errorgen_dual('S', a, normalization_factor='auto_return')
        self.assertArraysAlmostEqual(
            dual_eg * norm, expected)
        sparse = sps.csr_matrix(a)
        spL = lt.create_elementary_errorgen_dual('S', sparse, sparse=True)
        self.assertArraysAlmostEqual(spL.toarray() * norm, expected)

    def test_nonham_lindbladian(self):
        a = np.array([[1, 2], [3, 4]], 'd')
        b = np.array([[1, 2], [3, 4]], 'd')
        expected = np.array([
            [ -9,  -5,  -5,  4],
            [ -4, -11,   6,  1],
            [ -4,   6, -11,  1],
            [  9,   5,   5, -4]
        ], 'd')
        self.assertArraysAlmostEqual(lt.create_lindbladian_term_errorgen('O', a, b), expected)
        sparsea = sps.csr_matrix(a)
        sparseb = sps.csr_matrix(b)
        spL = lt.create_lindbladian_term_errorgen('O', sparsea, sparseb, sparse=True)
        self.assertArraysAlmostEqual(spL.toarray(),
                                     expected)

    def test_elementary_errorgen_bases(self):

        bases = [Basis.cast('gm', 4),
                 Basis.cast('pp', 4),
                 Basis.cast('PP', 4)]

        for basis in bases:
            print(basis)

            primals = []; duals = []; lbls = []
            for lbl, bel in zip(basis.labels[1:], basis.elements[1:]):
                lbls.append("H_%s" % lbl)
                primals.append(lt.create_elementary_errorgen('H', bel)) 
                duals.append(lt.create_elementary_errorgen_dual('H', bel)) 
            for lbl, bel in zip(basis.labels[1:], basis.elements[1:]):
                lbls.append("S_%s" % lbl)
                primals.append(lt.create_elementary_errorgen('S', bel))
                duals.append(lt.create_elementary_errorgen_dual('S', bel)) 
            for i, (lbl, bel) in enumerate(zip(basis.labels[1:], basis.elements[1:])):
                for lbl2, bel2 in zip(basis.labels[1+i+1:], basis.elements[1+i+1:]):
                    lbls.append("C_%s_%s" % (lbl, lbl2))
                    primals.append(lt.create_elementary_errorgen('C', bel, bel2))
                    duals.append(lt.create_elementary_errorgen_dual('C', bel, bel2)) 
            for i, (lbl, bel) in enumerate(zip(basis.labels[1:], basis.elements[1:])):
                for lbl2, bel2 in zip(basis.labels[1+i+1:], basis.elements[1+i+1:]):
                    lbls.append("A_%s_%s" % (lbl, lbl2))
                    primals.append(lt.create_elementary_errorgen('A', bel, bel2))
                    duals.append(lt.create_elementary_errorgen_dual('A', bel, bel2)) 

            dot_mx = np.empty((len(duals), len(primals)), complex)
            for i, dual in enumerate(duals):
                for j, primal in enumerate(primals):
                    dot_mx[i,j] = np.vdot(dual.flatten(), primal.flatten())

            self.assertTrue(np.allclose(dot_mx, np.identity(len(lbls), 'd')))

class RandomErrorgenRatesTester(BaseCase):

    def test_default_settings(self):
        random_errorgen_rates = lt.random_error_generator_rates(num_qubits=2, seed=1234, label_type='local')

        #make sure that we get the expected number of rates:
        self.assertEqual(len(random_errorgen_rates), 240)

        #also make sure this is CPTP, do so by constructing an error generator and confirming it doesn't fail
        #with CPTP parameterization. This should fail if the error generator dictionary is not CPTP.
        errorgen = LindbladErrorgen.from_elementary_errorgens(random_errorgen_rates, parameterization='CPTPLND', truncate=False, state_space=QubitSpace(2))

    def test_sector_restrictions(self):
        #H-only:
        random_errorgen_rates = lt.random_error_generator_rates(num_qubits=2, errorgen_types=('H',), seed=1234)
        #make sure that we get the expected number of rates:
        self.assertEqual(len(random_errorgen_rates), 15)
        #also make sure this is CPTP, do so by constructing an error generator and confirming it doesn't fail
        #with CPTP parameterization. This should fail if the error generator dictionary is not CPTP.
        errorgen = LindbladErrorgen.from_elementary_errorgens(random_errorgen_rates, parameterization='CPTPLND', truncate=False, state_space=QubitSpace(2))

        #S-only
        random_errorgen_rates = lt.random_error_generator_rates(num_qubits=2, errorgen_types=('S',), seed=1234)
        #make sure that we get the expected number of rates:
        self.assertEqual(len(random_errorgen_rates), 15)
        #also make sure this is CPTP, do so by constructing an error generator and confirming it doesn't fail
        #with CPTP parameterization. This should fail if the error generator dictionary is not CPTP.
        errorgen = LindbladErrorgen.from_elementary_errorgens(random_errorgen_rates, parameterization='CPTPLND', truncate=False, state_space=QubitSpace(2))

        #H+S
        random_errorgen_rates = lt.random_error_generator_rates(num_qubits=2, errorgen_types=('H','S'), seed=1234)
        #make sure that we get the expected number of rates:
        self.assertEqual(len(random_errorgen_rates), 30)
        #also make sure this is CPTP, do so by constructing an error generator and confirming it doesn't fail
        #with CPTP parameterization. This should fail if the error generator dictionary is not CPTP.
        errorgen = LindbladErrorgen.from_elementary_errorgens(random_errorgen_rates, parameterization='CPTPLND', truncate=False, state_space=QubitSpace(2))

        #H + S + A
        random_errorgen_rates = lt.random_error_generator_rates(num_qubits=2, errorgen_types=('H','S', 'A'), seed=1234)
        #make sure that we get the expected number of rates:
        self.assertEqual(len(random_errorgen_rates), 135)
        #also make sure this is CPTP, do so by constructing an error generator and confirming it doesn't fail
        #with CPTP parameterization. This should fail if the error generator dictionary is not CPTP.
        errorgen = LindbladErrorgen.from_elementary_errorgens(random_errorgen_rates, parameterization='CPTPLND', truncate=False, state_space=QubitSpace(2))

    def test_error_metric_restrictions(self):
        #test generator_infidelity
        random_errorgen_rates = lt.random_error_generator_rates(num_qubits=2, errorgen_types=('H','S'), 
                                                                error_metric= 'generator_infidelity', 
                                                                error_metric_value=.99, seed=1234)
        #confirm this has the correct generator infidelity.
        gen_infdl = 0
        for coeff, rate in random_errorgen_rates.items():
            if coeff.errorgen_type == 'H':
                gen_infdl+=rate**2
            elif coeff.errorgen_type == 'S':
                gen_infdl+=rate
        
        assert abs(gen_infdl-.99)<1e-5

        #test generator_error
        random_errorgen_rates = lt.random_error_generator_rates(num_qubits=2, errorgen_types=('H','S'), 
                                                                error_metric= 'total_generator_error', 
                                                                error_metric_value=.99, seed=1234)
        #confirm this has the correct generator infidelity.
        gen_error = 0
        for coeff, rate in random_errorgen_rates.items():
            if coeff.errorgen_type == 'H':
                gen_error+=abs(rate)
            elif coeff.errorgen_type == 'S':
                gen_error+=rate
        
        assert abs(gen_error-.99)<1e-5

        #test relative_HS_contribution:
        random_errorgen_rates = lt.random_error_generator_rates(num_qubits=2, errorgen_types=('H','S'), 
                                                                error_metric= 'generator_infidelity', 
                                                                error_metric_value=.99, 
                                                                relative_HS_contribution=(.5, .5), seed=1234)
        #confirm this has the correct generator infidelity contributions.
        gen_infdl_H = 0
        gen_infdl_S = 0
        for coeff, rate in random_errorgen_rates.items():
            if coeff.errorgen_type == 'H':
                gen_infdl_H+=rate**2
            elif coeff.errorgen_type == 'S':
                gen_infdl_S+=rate
        
        assert abs(gen_infdl_S - gen_infdl_H)<1e-5

        random_errorgen_rates = lt.random_error_generator_rates(num_qubits=2, errorgen_types=('H','S'), 
                                                                error_metric= 'total_generator_error', 
                                                                error_metric_value=.99, 
                                                                relative_HS_contribution=(.5, .5), seed=1234)
        #confirm this has the correct generator error contributions.
        gen_error_H = 0
        gen_error_S = 0
        for coeff, rate in random_errorgen_rates.items():
            if coeff.errorgen_type == 'H':
                gen_error_H+=abs(rate)
            elif coeff.errorgen_type == 'S':
                gen_error_S+=rate
        
        assert abs(gen_error_S - gen_error_H)<1e-5

    def test_fixed_errorgen_rates(self):
        fixed_rates_dict = {GlobalElementaryErrorgenLabel('H', ('X',), (0,)): 1}
        random_errorgen_rates = lt.random_error_generator_rates(num_qubits=2, errorgen_types=('H','S'), 
                                                                fixed_errorgen_rates=fixed_rates_dict, 
                                                                seed=1234)
        
        self.assertEqual(random_errorgen_rates[GlobalElementaryErrorgenLabel('H', ('X',), (0,))], 1)

    def test_label_type(self):

        random_errorgen_rates = lt.random_error_generator_rates(num_qubits=2, errorgen_types=('H','S'), 
                                                                label_type='local', seed=1234)
        assert isinstance(next(iter(random_errorgen_rates)), LocalElementaryErrorgenLabel)
        
    def test_sslbl_overlap(self):
        random_errorgen_rates = lt.random_error_generator_rates(num_qubits=2, errorgen_types=('H','S'), 
                                                                sslbl_overlap=(0,), 
                                                                seed=1234)
        for coeff in random_errorgen_rates:
            assert 0 in coeff.sslbls

