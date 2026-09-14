import warnings

import numpy as np
from scipy.linalg import expm

import pygsti
import pygsti.algorithms.cgstdesign as cd
import pygsti.tools.chartools as ct
from pygsti.algorithms import germselection as gs
from pygsti.circuits import Circuit
from pygsti.models.modelconstruction import create_explicit_model
from pygsti.processors import QubitProcessorSpec
from pygsti.protocols.cgst import (CharacterGSTDesign, CharacterGSTGermDesign,
                                   _select_fiducials, create_1q_szy_cgst_design)

from ..util import BaseCase, with_temp_path

_sx = np.array([[0, 1], [1, 0]], complex)


def _model(gate_names=('Gzpi2', 'Gypi2'), nonstd=None):
    pspec = QubitProcessorSpec(1, list(gate_names), nonstd_gate_unitaries=nonstd,
                               qubit_labels=('Q0',))
    return create_explicit_model(pspec, simulator='matrix')


def _circ(*gate_names):
    return Circuit([(name, 'Q0') for name in gate_names], line_labels=('Q0',))


class CandidateGermTester(BaseCase):

    def test_all_candidates_have_finite_order(self):
        mdl = _model()
        candidates = cd.finite_order_candidate_germs(mdl, 5)
        self.assertTrue(len(candidates) > 0)
        for germ in candidates:
            self.assertLessEqual(len(germ), 5)
            order = ct.germ_group_order(mdl.sim.product(germ), max_order=24)
            self.assertTrue(1 <= order <= 24)
        # {Gzpi2, Gypi2} generates the (finite) 1Q Clifford group, so nothing is dropped
        from pygsti.circuits import circuitconstruction as _cc
        allc = _cc.list_all_circuits_without_powers_and_cycles(list(mdl.primitive_op_labels), 5)
        self.assertEqual(len(candidates), len(allc))

    def test_infinite_order_gates_excluded(self):
        # a rotation by 1 radian has infinite order
        mdl = _model(('Gzpi2', 'Grot1'), nonstd={'Grot1': expm(-0.5j * 1.0 * _sx)})
        rot = _circ('Grot1')
        with self.assertRaises(ValueError):
            ct.germ_group_order(mdl.sim.product(rot), max_order=24)
        candidates = cd.finite_order_candidate_germs(mdl, 3)
        self.assertTrue(len(candidates) > 0)
        self.assertNotIn(rot, candidates)
        self.assertIn(_circ('Gzpi2'), candidates)
        for germ in candidates:
            ct.germ_group_order(mdl.sim.product(germ), max_order=24)  # must not raise

    def test_max_order_cutoff(self):
        mdl = _model()
        few = cd.finite_order_candidate_germs(mdl, 5, max_order=3)
        many = cd.finite_order_candidate_germs(mdl, 5, max_order=24)
        self.assertTrue(set(few).issubset(set(many)))
        self.assertTrue(len(few) < len(many))
        for germ in few:
            self.assertLessEqual(ct.germ_group_order(mdl.sim.product(germ), max_order=3), 3)

    def test_qubit_labels_override(self):
        mdl = _model()
        candidates = cd.finite_order_candidate_germs(mdl, 2, qubit_labels=('Q0',))
        for germ in candidates:
            self.assertEqual(germ.line_labels, ('Q0',))


class FindGermsTester(BaseCase):

    def test_amplificationally_complete_and_finite_order(self):
        mdl = _model()
        germs = cd.find_cgst_germs(mdl, max_length=7, seed=1)
        self.assertEqual(len(germs), 5)
        orders = [ct.germ_group_order(mdl.sim.product(g), max_order=24) for g in germs]
        # S, Y, SY and two order-3 compound germs (a length-6 and a length-4 one);
        # no order-2 germ, so every block of this design is analyzable
        self.assertEqual(orders, [4, 4, 3, 3, 3])
        self.assertEqual(max(len(g) for g in germs), 6)

        # amplificationally complete on a unitary-randomized 'full TP' copy
        tp = mdl.copy(); tp.set_all_parameterizations('full TP')
        rnd = tp.randomize_with_unitary(1e-3, seed=2)
        rnd.set_all_parameterizations('full TP')
        self.assertTrue(gs.test_germ_set_infl(rnd, germs))
        self.assertFalse(gs.test_germ_set_infl(rnd, germs[:2]))  # singletons alone are not AC

    def test_no_amplificationally_complete_set_raises(self):
        # force='singletons' is expanded against the FINITE-ORDER candidates, so
        # the 1-radian rotation is never forced into the germ set -- and no
        # finite-order germ set of length <= 4 is amplificationally complete for
        # this gate set, so the search must fail loudly (germselection itself just
        # prints a warning and returns an empty list) with an actionable message.
        mdl = _model(('Gzpi2', 'Grot1'), nonstd={'Grot1': expm(-0.5j * 1.0 * _sx)})
        with self.assertRaises(ValueError) as cm:
            cd.find_cgst_germs(mdl, max_length=4, seed=0)
        msg = str(cm.exception)
        self.assertIn('max_length', msg)
        self.assertIn('max_order', msg)
        self.assertIn('candidate', msg)

    def test_no_finite_order_candidates(self):
        mdl = _model(('Grot1',), nonstd={'Grot1': expm(-0.5j * 1.0 * _sx)})
        with self.assertRaises(ValueError):
            cd.find_cgst_germs(mdl, max_length=2)

    def test_order_one_germs_are_excluded(self):
        # a bare idle generates the trivial group: not a cGST germ; but germs that
        # *contain* the idle (and have finite order >= 2) are kept, and they are
        # not deduplicated away against their idle-free twins
        mdl = _model(('Gzpi2', 'Gypi2', 'Gi'))
        candidates = cd.finite_order_candidate_germs(mdl, 3)
        self.assertNotIn(_circ('Gi'), candidates)
        self.assertIn(_circ('Gzpi2', 'Gi'), candidates)
        self.assertIn(_circ('Gzpi2'), candidates)
        for germ in candidates:
            self.assertGreaterEqual(ct.germ_group_order(mdl.sim.product(germ), max_order=24), 2)
        deduped = cd._dedupe_candidate_germs(mdl, candidates)
        self.assertIn(_circ('Gzpi2', 'Gi'), deduped)
        self.assertIn(_circ('Gzpi2'), deduped)

    def test_idle_gate_set_yields_an_ac_germ_set(self):
        # the {S, sqrt(Y)} gate set *with* its idle.  pyGSTi's own find_germs fails
        # here because its superoperator-only deduplication discards every germ
        # that amplifies the idle's errors; the cGST search must not.
        mdl = _model(('Gzpi2', 'Gypi2', 'Gi'))
        germs = cd.find_cgst_germs(mdl, max_length=7, seed=0)
        self.assertTrue(any(any(lbl.name == 'Gi' for lbl in g.layertup) for g in germs))
        self.assertNotIn(_circ('Gi'), germs)
        for germ in germs:
            self.assertGreaterEqual(ct.germ_group_order(mdl.sim.product(germ), max_order=24), 2)
        tp = mdl.copy(); tp.set_all_parameterizations('full TP')
        rnd = tp.randomize_with_unitary(1e-3, seed=2)
        rnd.set_all_parameterizations('full TP')
        self.assertTrue(gs.test_germ_set_infl(rnd, germs))


class GermIrrepTester(BaseCase):

    def setUp(self):
        self.mdl = _model()

    def test_known_multiplicities(self):
        self.assertEqual(cd.germ_irreps(self.mdl, _circ('Gzpi2')), (4, {0: 2, 1: 1, 3: 1}))
        self.assertEqual(cd.germ_irreps(self.mdl, _circ('Gypi2', 'Gzpi2')), (3, {0: 2, 1: 1, 2: 1}))
        self.assertEqual(cd.germ_irreps(self.mdl, _circ('Gypi2', 'Gzpi2', 'Gzpi2')), (2, {0: 2, 1: 2}))

    def test_nonconjugate_irreps(self):
        self.assertEqual(cd.nonconjugate_irreps(4, {0: 2, 1: 1, 3: 1}), [0, 1])
        self.assertEqual(cd.nonconjugate_irreps(3, {0: 2, 1: 1, 2: 1}), [0, 1])
        self.assertEqual(cd.nonconjugate_irreps(2, {0: 2, 1: 2}), [0, 1])
        # a self-conjugate irrep (order/2) is kept; irreps the germ misses are dropped
        self.assertEqual(cd.nonconjugate_irreps(6, {0: 2, 2: 1, 3: 1, 4: 1}), [0, 2, 3])

    def test_multiplicities_sum_to_dimension(self):
        for germ in cd.finite_order_candidate_germs(self.mdl, 4):
            order, mult = cd.germ_irreps(self.mdl, germ)
            self.assertEqual(sum(mult.values()), self.mdl.dim)
            self.assertTrue(all(0 <= j < order for j in mult))


class FiducialSelectionTester(BaseCase):

    def test_reproduces_szy_fiducials(self):
        """The general selector must agree with cgst._select_fiducials for the {S, sqrt(Y)} germs."""
        mdl = _model()
        mdl_idle = _model(('Gzpi2', 'Gypi2', 'Gi'))
        specs = [(_circ('Gzpi2'), 4, 0), (_circ('Gzpi2'), 4, 1),
                 (_circ('Gypi2'), 4, 0), (_circ('Gypi2'), 4, 1),
                 (_circ('Gzpi2', 'Gypi2'), 3, 0), (_circ('Gzpi2', 'Gypi2'), 3, 1),
                 (_circ('Gzpi2', 'Gi'), 4, 1), (_circ('Gypi2', 'Gi'), 4, 1),
                 (_circ('Gzpi2', 'Gypi2', 'Gi'), 3, 1)]
        for germ, order, irrep in specs:
            old = _select_fiducials(germ, order, irrep, ('Gzpi2', 'Gypi2'), 'Q0')
            model = mdl_idle if any(lbl.name == 'Gi' for lbl in germ.layertup) else mdl
            new = cd.select_cgst_fiducials(model, germ, order, irrep)
            self.assertEqual(len(new), 1)
            self.assertEqual(new[0], old)

    def test_selected_pair_has_nonzero_amplitude(self):
        mdl = _model()
        germ, order, irrep = _circ('Gzpi2'), 4, 1
        (prep, meas), = cd.select_cgst_fiducials(mdl, germ, order, irrep)
        amp = cd._fiducial_overlap(mdl.sim.product(germ), order, irrep,
                                   mdl.sim.product(prep), mdl.sim.product(meas),
                                   *cd._model_spam(mdl))
        self.assertGreater(amp, 0.2)

    def test_unoccupied_irrep_gives_no_pairs(self):
        # an irrep with no decaying directions (here: one the germ doesn't
        # occupy at all) yields an empty pair list rather than a bogus pair
        mdl = _model()
        germ = _circ('Gzpi2')
        _, mult = cd.germ_irreps(mdl, germ)
        self.assertEqual(mult.get(2, 0), 0)
        self.assertEqual(cd.select_cgst_fiducials(mdl, germ, 4, 2), [])

    def test_degenerate_block(self):
        mdl = _model()
        germ = _circ('Gypi2', 'Gzpi2', 'Gzpi2')  # order 2, irrep 1 has multiplicity 2
        order, mult = cd.germ_irreps(mdl, germ)
        self.assertEqual((order, mult[1]), (2, 2))
        pairs = cd.select_cgst_fiducials(mdl, germ, order, 1)
        self.assertEqual(len(pairs), 4)
        preps = sorted(set(p for p, _ in pairs), key=str)
        meas = sorted(set(m for _, m in pairs), key=str)
        self.assertEqual((len(preps), len(meas)), (2, 2))

        # the block-projected overlap matrices must be full rank
        rho, evec, ident_vec = cd._model_spam(mdl)
        proj = cd._decaying_projector(mdl.sim.product(germ), order, 1, ident_vec)
        u, s, _ = np.linalg.svd(proj)
        block = u[:, :2]
        pmat = np.array([block.conj().T @ (mdl.sim.product(p) @ rho) for p in preps])
        mmat = np.array([(evec @ mdl.sim.product(m)) @ block for m in meas])
        self.assertEqual(np.linalg.matrix_rank(pmat, tol=1e-8), 2)
        self.assertEqual(np.linalg.matrix_rank(mmat, tol=1e-8), 2)

    def test_num_pairs_override(self):
        mdl = _model()
        germ = _circ('Gypi2', 'Gzpi2', 'Gzpi2')
        self.assertEqual(len(cd.select_cgst_fiducials(mdl, germ, 2, 1, num_pairs=2)), 2)
        self.assertEqual(len(cd.select_cgst_fiducials(mdl, germ, 2, 1, num_pairs=1)), 1)

    def test_too_few_fiducial_words_for_a_degenerate_block(self):
        # a 2-dimensional block needs 2 distinct words per side; with max_length=0
        # only the empty word exists, which must be a clear error (not a TypeError)
        mdl = _model()
        germ = _circ('Gypi2', 'Gzpi2', 'Gzpi2')
        with self.assertRaises(ValueError) as cm:
            cd.select_cgst_fiducials(mdl, germ, 2, 1, max_length=0)
        self.assertIn('max_length', str(cm.exception))


class GermNameTester(BaseCase):

    def test_single_qubit_names_omit_qubit_labels(self):
        self.assertEqual(cd.germ_name(_circ('Gzpi2', 'Gypi2')), 'Gzpi2Gypi2')
        self.assertEqual(cd.germ_name(_circ('Gzpi2')), 'Gzpi2')

    def test_multi_qubit_names_carry_qubit_labels(self):
        germ = Circuit([('Gxpi2', 'Q0'), ('Gcnot', 'Q0', 'Q1')], line_labels=('Q0', 'Q1'))
        self.assertEqual(cd.germ_name(germ), 'Gxpi2Q0GcnotQ0Q1')
        other = Circuit([('Gxpi2', 'Q1'), ('Gcnot', 'Q0', 'Q1')], line_labels=('Q0', 'Q1'))
        self.assertNotEqual(cd.germ_name(germ), cd.germ_name(other))
        parallel = Circuit([[('Gxpi2', 'Q0'), ('Gypi2', 'Q1')]], line_labels=('Q0', 'Q1'))
        self.assertEqual(cd.germ_name(parallel), 'Gxpi2Q0-Gypi2Q1')
        for name in (cd.germ_name(germ), cd.germ_name(parallel)):
            self.assertTrue(all(ch.isalnum() or ch in '_-' for ch in name))


class CreateDesignTester(BaseCase):

    def _check_design(self, mdl, design):
        self.assertIsInstance(design, CharacterGSTDesign)
        self.assertEqual(len(set(design.keys())), len(design.keys()))  # unique names
        self.assertEqual(len(design.germ_table), len(design.keys()))
        for row in design.germ_table:
            child = design[row['name']]
            self.assertIsInstance(child, CharacterGSTGermDesign)
            self.assertEqual(child.germ.str, row['germ'])
            self.assertEqual(child.group_order, row['group_order'])
            self.assertEqual(child.irrep_index, row['irrep_index'])
            self.assertEqual(child.prep_fiducial.str, row['prep_fiducial'])
            self.assertEqual(child.meas_fiducial.str, row['meas_fiducial'])
            order, mult = cd.germ_irreps(mdl, child.germ)
            self.assertEqual(order, row['group_order'])
            self.assertEqual(mult[row['irrep_index']], row['multiplicity'])
            self.assertTrue(isinstance(row['name'], str))
            self.assertTrue(all(isinstance(v, (str, int)) for v in row.values()))

    def test_zy_design(self):
        mdl = _model()
        design = cd.create_cgst_design(mdl, [0, 1, 2], 4, mode='exact', seed=3)
        self._check_design(mdl, design)
        germs = set(row['germ'] for row in design.germ_table)
        self.assertEqual(len(germs), 5)
        self.assertTrue(len(design.all_circuits_needing_data) > 0)

    def test_xy_design(self):
        mdl = _model(('Gxpi2', 'Gypi2'))
        design = cd.create_cgst_design(mdl, [0, 1], 4, mode='exact', seed=5)
        self._check_design(mdl, design)
        for name in design.keys():
            self.assertTrue(name.startswith('Gxpi2') or name.startswith('Gypi2'))

    def test_explicit_germs_and_irrep_choices(self):
        mdl = _model()
        germs = [_circ('Gzpi2'), _circ('Gzpi2', 'Gypi2')]
        nonconj = cd.create_cgst_design(mdl, [0, 1], 4, germs=germs, mode='exact', seed=0)
        self.assertEqual(len(nonconj.keys()), 4)  # 2 germs x (trivial + 1 conjugate rep)
        alli = cd.create_cgst_design(mdl, [0, 1], 4, germs=germs, mode='exact',
                                     irreps='all', seed=0)
        self.assertEqual(len(alli.keys()), 6)  # 3 occupied irreps per germ
        explicit = cd.create_cgst_design(mdl, [0, 1], 4, germs=germs, mode='exact',
                                         irreps={germs[0]: [1], germs[1]: [0]}, seed=0)
        self.assertEqual(sorted(explicit.keys()), ['Gzpi2Gypi2_irrep0_pair0', 'Gzpi2_irrep1_pair0'])
        # names must be safe as directory names on every platform (no ':')
        for name in list(nonconj.keys()) + list(alli.keys()):
            self.assertTrue(all(ch.isalnum() or ch in '_-' for ch in name), name)

    def test_default_mode_is_exact(self):
        mdl = _model()
        design = cd.create_cgst_design(mdl, [0, 1], 4, germs=[_circ('Gzpi2')], seed=0)
        self.assertTrue(all(design[name].mode == 'exact' for name in design.keys()))

    def test_order_one_germ_is_skipped_with_warning(self):
        mdl = _model(('Gzpi2', 'Gypi2', 'Gi'))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            design = cd.create_cgst_design(mdl, [0, 1], 4, germs=[_circ('Gi'), _circ('Gzpi2')],
                                           mode='exact', seed=0)
        self.assertTrue(any('group order 1' in str(w.message) for w in caught))
        self.assertEqual(set(row['germ'] for row in design.germ_table), {_circ('Gzpi2').str})
        with self.assertRaises(ValueError):
            cd.create_cgst_design(mdl, [0, 1], 4, germs=[_circ('Gi')], mode='exact', seed=0)

    def test_include_degenerate_blocks(self):
        mdl = _model()
        germs = [_circ('Gzpi2'), _circ('Gypi2', 'Gzpi2', 'Gzpi2')]  # the latter: order 2, both blocks 2-dim
        full = cd.create_cgst_design(mdl, [0, 1], 4, germs=germs, mode='exact', seed=0)
        lean = cd.create_cgst_design(mdl, [0, 1], 4, germs=germs, mode='exact', seed=0,
                                     include_degenerate_blocks=False)
        # S: trivial (mult 2) + irrep 1 (mult 1); pi rotation: trivial (mult 2) + 2x2 grid on irrep 1
        self.assertEqual(len(full.keys()), 2 + 1 + 4)
        self.assertEqual(len(lean.keys()), 2 + 1)
        self.assertTrue(set(lean.keys()).issubset(set(full.keys())))
        for row in lean.germ_table:
            if row['irrep_index'] == 0:
                self.assertEqual(row['multiplicity'], 2)
            else:
                self.assertEqual(row['multiplicity'], 1)

    def test_distinct_child_seeds(self):
        mdl = _model()
        germs = [_circ('Gzpi2'), _circ('Gypi2')]
        design = cd.create_cgst_design(mdl, [1, 2], 8, germs=germs, mode='reduced', seed=11)
        exponents = [tuple(map(tuple, design[name].exponent_lists[0])) for name in design.keys()]
        self.assertTrue(len(set(exponents)) > 1)

    def test_bad_irreps_argument(self):
        mdl = _model()
        with self.assertRaises(ValueError):
            cd.create_cgst_design(mdl, [0, 1], 4, germs=[_circ('Gzpi2')], irreps='bogus')

    @with_temp_path
    def test_serialization_roundtrip(self, pth):
        mdl = _model()
        design = cd.create_cgst_design(mdl, [0, 1], 4, germs=[_circ('Gzpi2'), _circ('Gzpi2', 'Gypi2')],
                                       mode='exact', seed=9)
        design.write(pth)
        loaded = pygsti.io.read_edesign_from_dir(pth)
        self.assertIsInstance(loaded, CharacterGSTDesign)
        self.assertEqual(set(loaded.keys()), set(design.keys()))
        self.assertEqual(loaded.germ_table, design.germ_table)
        for name in design.keys():
            self.assertEqual(list(loaded[name].all_circuits_needing_data),
                             list(design[name].all_circuits_needing_data))

    @with_temp_path
    def test_germ_table_none_roundtrip(self, pth):
        design = create_1q_szy_cgst_design([0, 1], 4, include_idle=False, seed=2)
        self.assertIsNone(design.germ_table)
        design.write(pth)
        loaded = pygsti.io.read_edesign_from_dir(pth)
        self.assertIsNone(loaded.germ_table)
