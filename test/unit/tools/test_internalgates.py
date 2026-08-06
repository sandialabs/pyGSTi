import numpy as np

# from pygsti.extras import rb
from pygsti.tools import internalgates, optools as ot, basistools as bt
from ..util import BaseCase


class InternalGatesTester(BaseCase):

    def test_internalgate_definitions(self):
        # TODO is this test needed?
        self.skipTest("RB analysis is known to be broken.  Skip tests until it gets fixed.")
        std_unitaries = internalgates.standard_gatename_unitaries()
        std_quil = internalgates.standard_gatenames_quil_conversions()
        std_quil = internalgates.standard_gatenames_openqasm_conversions()

        # Checks the standard Clifford gate unitaries agree with the Clifford group unitaries.
        group = rb.group.construct_1q_clifford_group()
        for key in group.labels:
            self.assertLess(np.sum(abs(np.array(group.matrix(key))
                                       - ot.unitary_to_pauligate(std_unitaries[key]))), 10**-10)

    def test_u3_unitary_generator(self):
        # Checks the u3 unitary generator runs
        u = internalgates.qasm_u3(0., 0., 0., output='unitary')
        sup = internalgates.qasm_u3(0., 0., 0., output='superoperator')
        sup_u = ot.std_process_mx_to_unitary(bt.change_basis(sup, 'pp', 'std')) # Backtransform to unitary
        self.assertArraysAlmostEqual(u, sup_u)


def _unitaries_equal_up_to_global_phase(U1, U2, tol=1e-8):
    """Helper: True if U1 == c*U2 for some scalar c with |c| == 1."""
    U1 = np.asarray(U1)
    U2 = np.asarray(U2)
    if U1.shape != U2.shape:
        return False
    # locate an entry of U2 with the largest magnitude to safely divide by.
    idx = np.unravel_index(np.argmax(np.abs(U2)), U2.shape)
    if np.abs(U2[idx]) < tol:
        return np.allclose(U1, 0, atol=tol)
    phase = U1[idx] / U2[idx]
    if not np.isclose(np.abs(phase), 1.0, atol=tol):
        return False
    return np.allclose(U1, phase * U2, atol=tol)


class InternalGatesCirqConversionTester(BaseCase):
    """Tests for the cirq <-> pyGSTi conversions used by ``Circuit.from_cirq``/``convert_to_cirq``."""

    def setUp(self):
        try:
            import cirq  # noqa: F401
        except ImportError:
            self.skipTest("Cirq is required for this operation, and it does not appear to be installed.")

    def test_parameterized_gate_unitary_conventions(self):
        # For a grid of angles, check that pyGSTi's parameterized standard-gate unitaries
        # agree (up to global phase) with cirq's unitary for the corresponding pow-gate.
        import cirq

        std_unitaries = internalgates.standard_gatename_unitaries()
        angles = np.linspace(-2 * np.pi, 2 * np.pi, 9)

        for theta in angles:
            t = theta / np.pi

            # Gzr/Gczr are exact matches (no global phase ambiguity), since global_shift=0
            # pins the (0,0) matrix entry to 1 for these diagonal gates.
            self.assertArraysAlmostEqual(std_unitaries['Gzr']([theta]), cirq.unitary(cirq.ZPowGate(exponent=t)))
            self.assertArraysAlmostEqual(std_unitaries['Gczr']([theta]), cirq.unitary(cirq.CZPowGate(exponent=t)))

            # Gxr/Gyr match cirq's X^t/Y^t only up to a global phase.
            self.assertTrue(_unitaries_equal_up_to_global_phase(
                std_unitaries['Gxr']([theta]), cirq.unitary(cirq.XPowGate(exponent=t))))
            self.assertTrue(_unitaries_equal_up_to_global_phase(
                std_unitaries['Gyr']([theta]), cirq.unitary(cirq.YPowGate(exponent=t))))

    def test_phasedxz_to_gu3_conventions(self):
        # Check the PhasedXZGate -> Gu3 args conversion against cirq's unitary, up to global phase,
        # for a handful of arbitrary (non-Clifford) angle triples.
        import cirq

        std_unitaries = internalgates.standard_gatename_unitaries()
        family_conversions = dict((cls.__name__, (name, argfn)) for cls, name, argfn
                                  in internalgates.cirq_parameterized_gatenames_standard_conversions())
        _, gu3_args = family_conversions['PhasedXZGate']

        rng = np.random.RandomState(1234)
        for _ in range(10):
            a, x, z = rng.uniform(-1, 1, size=3)
            gate = cirq.PhasedXZGate(axis_phase_exponent=a, x_exponent=x, z_exponent=z)
            args = gu3_args(gate)
            self.assertTrue(_unitaries_equal_up_to_global_phase(
                std_unitaries['Gu3'](args), cirq.unitary(gate)))

    def test_cirq_parameterized_gatenames_standard_conversions_dispatch(self):
        # Rz/Rx/Ry are implemented as subclasses of Z/X/YPowGate in cirq, so they should be
        # matched by the same isinstance-based family entries without needing dedicated ones.
        import cirq

        family_conversions = internalgates.cirq_parameterized_gatenames_standard_conversions()

        def _dispatch(gate):
            for cls, name, argfn in family_conversions:
                if isinstance(gate, cls):
                    return name, argfn(gate)
            return None, None

        std_unitaries = internalgates.standard_gatename_unitaries()

        for gate in (cirq.rz(0.37), cirq.rx(0.71), cirq.ry(-0.42)):
            name, args = _dispatch(gate)
            self.assertIsNotNone(name)
            self.assertTrue(_unitaries_equal_up_to_global_phase(std_unitaries[name](args), cirq.unitary(gate)))

    def test_standard_gatenames_cirq_conversions_callables(self):
        # The parameterized gate families round-trip through the pyGSTi -> cirq conversion dict
        # as callables (angle args -> cirq gate), rather than a fixed cirq gate object.
        import cirq

        std_gatenames_to_cirq = internalgates.standard_gatenames_cirq_conversions()
        std_unitaries = internalgates.standard_gatename_unitaries()

        for name in ('Gzr', 'Gxr', 'Gyr', 'Gczr'):
            self.assertTrue(callable(std_gatenames_to_cirq[name]))

        theta = 0.53
        self.assertArraysAlmostEqual(cirq.unitary(std_gatenames_to_cirq['Gzr'](theta)), std_unitaries['Gzr']([theta]))
        self.assertTrue(_unitaries_equal_up_to_global_phase(
            cirq.unitary(std_gatenames_to_cirq['Gxr'](theta)), std_unitaries['Gxr']([theta])))
        self.assertTrue(_unitaries_equal_up_to_global_phase(
            cirq.unitary(std_gatenames_to_cirq['Gyr'](theta)), std_unitaries['Gyr']([theta])))
        self.assertArraysAlmostEqual(cirq.unitary(std_gatenames_to_cirq['Gczr'](theta)), std_unitaries['Gczr']([theta]))

        theta, phi, lamb = 0.4, -0.9, 1.7
        self.assertTrue(_unitaries_equal_up_to_global_phase(
            cirq.unitary(std_gatenames_to_cirq['Gu3'](theta, phi, lamb)), std_unitaries['Gu3']([theta, phi, lamb])))

    def test_cirq_gatenames_standard_conversions_unaffected(self):
        # Regression check: adding the new callable entries to standard_gatenames_cirq_conversions
        # must not break the (much larger) exact-match reverse mapping used by from_cirq, since
        # cirq.Gate instances are themselves callable and a naive `callable(value)` filter would
        # incorrectly drop them all.
        import cirq

        cirq_to_standard = internalgates.cirq_gatenames_standard_conversions()
        self.assertGreater(len(cirq_to_standard), 30)
        self.assertEqual(cirq_to_standard[cirq.CNOT], 'Gcnot')
        self.assertEqual(cirq_to_standard[cirq.H], 'Gh')
        self.assertFalse(any(name not in internalgates.standard_gatename_unitaries()
                             for name in cirq_to_standard.values()))
