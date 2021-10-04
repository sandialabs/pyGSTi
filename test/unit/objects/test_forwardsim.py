# XXX rewrite or remove

from unittest import mock

import numpy as np

import pygsti.models as models
from pygsti.forwardsims.forwardsim import ForwardSimulator
from pygsti.forwardsims.mapforwardsim import MapForwardSimulator
from pygsti.models import ExplicitOpModel
from pygsti.circuits import Circuit
from pygsti.baseobjs import Label as L
from ..util import BaseCase


def Ls(*args):
    """ Convert args to a tuple to Labels """
    return tuple([L(x) for x in args])


class AbstractForwardSimTester(BaseCase):
    # XXX is it really neccessary to test an abstract base class?
    def setUp(self):
        mock_model = mock.MagicMock()
        mock_model.evotype.return_value = "densitymx"
        mock_model.circuit_outcomes.return_value = ('NA',)
        mock_model.num_params = 0
        self.fwdsim = ForwardSimulator(mock_model)
        self.circuit = Circuit("GxGx")

    def test_create_layout(self):
        self.fwdsim.create_layout([self.circuit])

    def test_bulk_fill_probs(self):
        layout = self.fwdsim.create_layout([self.circuit])
        with self.assertRaises(NotImplementedError):
            self.fwdsim.bulk_fill_probs(np.zeros(1), layout)

    def test_bulk_fill_dprobs(self):
        layout = self.fwdsim.create_layout([self.circuit])
        with self.assertRaises(NotImplementedError):
            self.fwdsim.bulk_fill_dprobs(np.zeros((1,0)), layout)

    def test_bulk_fill_hprobs(self):
        layout = self.fwdsim.create_layout([self.circuit])
        with self.assertRaises(NotImplementedError):
            self.fwdsim.bulk_fill_hprobs(np.zeros((1,0,0)), layout)

#    def test_iter_hprobs_by_rectangle(self):
#        with self.assertRaises(NotImplementedError):
#            self.fwdsim.bulk_fill_hprobs(None, None)


class ForwardSimBase(object):
    @classmethod
    def setUpClass(cls):
        ExplicitOpModel._strict = False
        cls.model = models.create_explicit_model_from_expressions(
            [('Q0',)], ['Gi', 'Gx', 'Gy'],
            ["I(Q0)", "X(pi/8,Q0)", "Y(pi/8,Q0)"]
        )

    def setUp(self):
        self.fwdsim = self.model.sim
        self.layout = self.fwdsim.create_layout([('Gx',), ('Gx', 'Gx')], array_types=('e', 'ep', 'epp'))
        self.nP = self.model.num_params
        self.nEls = self.layout.num_elements

    def test_bulk_fill_probs(self):
        pmx = np.empty(self.nEls, 'd')
        print(self.fwdsim.model._opcaches)
        self.fwdsim.bulk_fill_probs(pmx, self.layout)
        # TODO assert correctness

    def test_bulk_fill_dprobs(self):
        dmx = np.empty((self.nEls, self.nP), 'd')
        pmx = np.empty(self.nEls, 'd')
        self.fwdsim.bulk_fill_dprobs(dmx, self.layout, pr_array_to_fill=pmx)
        # TODO assert correctness

    def test_bulk_fill_dprobs_with_block_size(self):
        dmx = np.empty((self.nEls, self.nP), 'd')
        self.fwdsim.bulk_fill_dprobs(dmx, self.layout)
        # TODO assert correctness

    def test_bulk_fill_hprobs(self):
        hmx = np.zeros((self.nEls, self.nP, self.nP), 'd')
        dmx = np.zeros((self.nEls, self.nP), 'd')
        pmx = np.zeros(self.nEls, 'd')
        self.fwdsim.bulk_fill_hprobs(hmx, self.layout,
                                     pr_array_to_fill=pmx, deriv1_array_to_fill=dmx, deriv2_array_to_fill=dmx)
        # TODO assert correctness

        hmx = np.zeros((self.nEls, self.nP, self.nP), 'd')
        dmx1 = np.zeros((self.nEls, self.nP), 'd')
        dmx2 = np.zeros((self.nEls, self.nP), 'd')
        pmx = np.zeros(self.nEls, 'd')
        self.fwdsim.bulk_fill_hprobs(hmx, self.layout,
                                     pr_array_to_fill=pmx, deriv1_array_to_fill=dmx1, deriv2_array_to_fill=dmx2)
        # TODO assert correctness

    def test_iter_hprobs_by_rectangle(self):
        # TODO optimize
        mx = np.zeros((self.nEls, self.nP, self.nP), 'd')
        dmx1 = np.zeros((self.nEls, self.nP), 'd')
        dmx2 = np.zeros((self.nEls, self.nP), 'd')
        pmx = np.zeros(self.nEls, 'd')
        self.fwdsim.bulk_fill_hprobs(mx, self.layout, pr_array_to_fill=pmx,
                                     deriv1_array_to_fill=dmx1, deriv2_array_to_fill=dmx2)
        # TODO assert correctness

    #REMOVE
    #def test_prs(self):
    #    
    #    self.fwdsim._prs(L('rho0'), [L('Mdefault_0')], Ls('Gx', 'Gx'), clip_to=(-1, 1))
    #    self.fwdsim._prs(L('rho0'), [L('Mdefault_0')], Ls('Gx', 'Gx'), clip_to=(-1, 1), use_scaling=True)
    #    # TODO assert correctness
    #
    #def test_estimate_cache_size(self):
    #    self.fwdsim._estimate_cache_size(100)
    #    # TODO assert correctness
    #
    #def test_estimate_mem_usage(self):
    #    est = self.fwdsim.estimate_memory_usage(
    #        ["bulk_fill_probs", "bulk_fill_dprobs", "bulk_fill_hprobs"],
    #        cache_size=100, num_subtrees=2, num_subtree_proc_groups=1,
    #        num_param1_groups=1, num_param2_groups=1, num_final_strs=100
    #    )
    #    # TODO assert correctness
    #
    #def test_estimate_mem_usage_raises_on_bad_subcall_key(self):
    #    with self.assertRaises(ValueError):
    #        self.fwdsim.estimate_memory_usage(["foobar"], 1, 1, 1, 1, 1, 1)


class MatrixForwardSimTester(ForwardSimBase, BaseCase):
    def test_doperation(self):
        dg = self.fwdsim._doperation(L('Gx'), flat=False)
        dgflat = self.fwdsim._doperation(L('Gx'), flat=True)
        # TODO assert correctness

    def test_hoperation(self):
        hg = self.fwdsim._hoperation(L('Gx'), flat=False)
        hgflat = self.fwdsim._hoperation(L('Gx'), flat=True)
        # TODO assert correctness

    #REMOVE
    #def test_hproduct(self):
    #    self.fwdsim.hproduct(Ls('Gx', 'Gx'), flat=True, wrt_filter1=[0, 1], wrt_filter2=[1, 2, 3])
    #    # TODO assert correctness
    #def test_hpr(self):
    #    self.fwdsim._hpr(Ls('rho0', 'Mdefault_0'), Ls('Gx', 'Gx'), False, False, clip_to=(-1, 1))
    #    # TODO assert correctness

    #TODO: we moved _dpr and _hpr from MatrixForwardSimulator to here.  Maybe they can be made into
    # unit tests?  These are for computing the derivative and hessian of a single circuit...
    #def _dpr(self, spam_tuple, circuit, return_pr, clip_to):
    #    """
    #    Compute the derivative of the probability corresponding to `circuit` and `spam_tuple`.
    #
    #    Parameters
    #    ----------
    #    spam_tuple : (rho_label, simplified_effect_label)
    #        Specifies the prep and POVM effect used to compute the probability.
    #
    #    circuit : Circuit or tuple
    #        A tuple-like object of *simplified* gates (e.g. may include
    #        instrument elements like 'Imyinst_0')
    #
    #    return_pr : bool
    #        when set to True, additionally return the probability itself.
    #
    #    clip_to : 2-tuple
    #        (min,max) to clip returned probability to if not None.
    #        Only relevant when pr_mx_to_fill is not None.
    #
    #    Returns
    #    -------
    #    derivative : numpy array
    #        a 1 x M numpy array of derivatives of the probability w.r.t.
    #        each model parameter (M is the number of model parameters).
    #
    #    probability : float
    #        only returned if return_pr == True.
    #    """
    #    if self.evotype == "statevec": raise NotImplementedError("Unitary evolution not fully supported yet!")
    #    # To support unitary evolution we need to:
    #    # - alter product, dproduct, etc. to allow for *complex* derivatives, since matrices can be complex
    #    # - update probability-deriv. computations: dpr/dx -> d|pr|^2/dx = d(pr*pr.C)/dx = dpr/dx*pr.C + pr*dpr/dx.C
    #    #    = 2 Re(dpr/dx*pr.C) , where dpr/dx is the usual density-matrix-mode probability
    #    # (TODO in FUTURE)
    #
    #    #  pr = Tr( |rho><E| * prod ) = sum E_k prod_kl rho_l
    #    #  dpr/d(op_label)_ij = sum E_k [dprod/d(op_label)_ij]_kl rho_l
    #    #  dpr/d(rho)_i = sum E_k prod_ki
    #    #  dpr/d(E)_i   = sum prod_il rho_l
    #
    #    rholabel, elabel = spam_tuple  # can't deal w/"custom" spam label...
    #    rho, E = self._rho_e_from_spam_tuple(spam_tuple)
    #    rhoVec = self.model.prep(rholabel)  # distinct from rho,E b/c rho,E are
    #    EVec = self.model.effect(elabel)   # arrays, these are SPAMVecs
    #
    #    #Derivs wrt Gates
    #    old_err = _np.seterr(over='ignore')
    #    prod, scale = self.product(circuit, True)
    #    dprod_dOps = self.dproduct(circuit)
    #    dpr_dOps = _np.empty((1, self.model.num_params))
    #    for i in range(self.model.num_params):
    #        dpr_dOps[0, i] = float(_np.dot(E, _np.dot(dprod_dOps[i], rho)))
    #
    #    if return_pr:
    #        p = _np.dot(E, _np.dot(prod, rho)) * scale  # may generate overflow, but OK
    #        if clip_to is not None: p = _np.clip(p, clip_to[0], clip_to[1])
    #
    #    #Derivs wrt SPAM
    #    derivWrtAnyRhovec = scale * _np.dot(E, prod)
    #    dpr_drhos = _np.zeros((1, self.model.num_params))
    #    _fas(dpr_drhos, [0, self.model.prep(rholabel).gpindices],
    #         _np.dot(derivWrtAnyRhovec, rhoVec.deriv_wrt_params()))  # may overflow, but OK
    #
    #    dpr_dEs = _np.zeros((1, self.model.num_params))
    #    derivWrtAnyEvec = scale * _np.transpose(_np.dot(prod, rho))  # may overflow, but OK
    #    # (** doesn't depend on eIndex **) -- TODO: should also conjugate() here if complex?
    #    _fas(dpr_dEs, [0, EVec.gpindices],
    #         _np.dot(derivWrtAnyEvec, EVec.deriv_wrt_params()))
    #
    #    _np.seterr(**old_err)
    #
    #    if return_pr:
    #        return dpr_drhos + dpr_dEs + dpr_dOps, p
    #    else: return dpr_drhos + dpr_dEs + dpr_dOps
    #
    #def _hpr(self, spam_tuple, circuit, return_pr, return_deriv, clip_to):
    #    """
    #    Compute the Hessian of the probability given by `circuit` and `spam_tuple`.
    #
    #    Parameters
    #    ----------
    #    spam_tuple : (rho_label, simplified_effect_label)
    #        Specifies the prep and POVM effect used to compute the probability.
    #
    #    circuit : Circuit or tuple
    #        A tuple-like object of *simplified* gates (e.g. may include
    #        instrument elements like 'Imyinst_0')
    #
    #    return_pr : bool
    #        when set to True, additionally return the probability itself.
    #
    #    return_deriv : bool
    #        when set to True, additionally return the derivative of the
    #        probability.
    #
    #    clip_to : 2-tuple
    #        (min,max) to clip returned probability to if not None.
    #        Only relevant when pr_mx_to_fill is not None.
    #
    #    Returns
    #    -------
    #    hessian : numpy array
    #        a 1 x M x M array, where M is the number of model parameters.
    #        hessian[0,j,k] is the derivative of the probability w.r.t. the
    #        k-th then the j-th model parameter.
    #
    #    derivative : numpy array
    #        only returned if return_deriv == True. A 1 x M numpy array of
    #        derivatives of the probability w.r.t. each model parameter.
    #
    #    probability : float
    #        only returned if return_pr == True.
    #    """
    #    if self.evotype == "statevec": raise NotImplementedError("Unitary evolution not fully supported yet!")
    #
    #    #  pr = Tr( |rho><E| * prod ) = sum E_k prod_kl rho_l
    #    #  d2pr/d(opLabel1)_mn d(opLabel2)_ij = sum E_k [dprod/d(opLabel1)_mn d(opLabel2)_ij]_kl rho_l
    #    #  d2pr/d(rho)_i d(op_label)_mn = sum E_k [dprod/d(op_label)_mn]_ki     (and same for other diff order)
    #    #  d2pr/d(E)_i d(op_label)_mn   = sum [dprod/d(op_label)_mn]_il rho_l   (and same for other diff order)
    #    #  d2pr/d(E)_i d(rho)_j          = prod_ij                                (and same for other diff order)
    #    #  d2pr/d(E)_i d(E)_j            = 0
    #    #  d2pr/d(rho)_i d(rho)_j        = 0
    #
    #    rholabel, elabel = spam_tuple
    #    rho, E = self._rho_e_from_spam_tuple(spam_tuple)
    #    rhoVec = self.model.prep(rholabel)  # distinct from rho,E b/c rho,E are
    #    EVec = self.model.effect(elabel)   # arrays, these are SPAMVecs
    #
    #    d2prod_dGates = self.hproduct(circuit)
    #    assert(d2prod_dGates.shape[0] == d2prod_dGates.shape[1])
    #
    #    d2pr_dOps2 = _np.empty((1, self.model.num_params, self.model.num_params))
    #    for i in range(self.model.num_params):
    #        for j in range(self.model.num_params):
    #            d2pr_dOps2[0, i, j] = float(_np.dot(E, _np.dot(d2prod_dGates[i, j], rho)))
    #
    #    old_err = _np.seterr(over='ignore')
    #
    #    prod, scale = self.product(circuit, True)
    #    if return_pr:
    #        p = _np.dot(E, _np.dot(prod, rho)) * scale  # may generate overflow, but OK
    #        if clip_to is not None: p = _np.clip(p, clip_to[0], clip_to[1])
    #
    #    dprod_dOps = self.dproduct(circuit)
    #    assert(dprod_dOps.shape[0] == self.model.num_params)
    #    if return_deriv:  # same as in dpr(...)
    #        dpr_dOps = _np.empty((1, self.model.num_params))
    #        for i in range(self.model.num_params):
    #            dpr_dOps[0, i] = float(_np.dot(E, _np.dot(dprod_dOps[i], rho)))
    #
    #    #Derivs wrt SPAM
    #    if return_deriv:  # same as in dpr(...)
    #        dpr_drhos = _np.zeros((1, self.model.num_params))
    #        derivWrtAnyRhovec = scale * _np.dot(E, prod)
    #        _fas(dpr_drhos, [0, self.model.prep(rholabel).gpindices],
    #             _np.dot(derivWrtAnyRhovec, rhoVec.deriv_wrt_params()))  # may overflow, but OK
    #
    #        dpr_dEs = _np.zeros((1, self.model.num_params))
    #        derivWrtAnyEvec = scale * _np.transpose(_np.dot(prod, rho))  # may overflow, but OK
    #        _fas(dpr_dEs, [0, EVec.gpindices],
    #             _np.dot(derivWrtAnyEvec, EVec.deriv_wrt_params()))
    #
    #        dpr = dpr_drhos + dpr_dEs + dpr_dOps
    #
    #    d2pr_drhos = _np.zeros((1, self.model.num_params, self.model.num_params))
    #    _fas(d2pr_drhos, [0, None, self.model.prep(rholabel).gpindices],
    #         _np.dot(_np.dot(E, dprod_dOps), rhoVec.deriv_wrt_params())[0])  # (= [0,:,:])
    #
    #    d2pr_dEs = _np.zeros((1, self.model.num_params, self.model.num_params))
    #    derivWrtAnyEvec = _np.squeeze(_np.dot(dprod_dOps, rho), axis=(2,))
    #    _fas(d2pr_dEs, [0, None, EVec.gpindices],
    #         _np.dot(derivWrtAnyEvec, EVec.deriv_wrt_params()))
    #
    #    d2pr_dErhos = _np.zeros((1, self.model.num_params, self.model.num_params))
    #    derivWrtAnyEvec = scale * _np.dot(prod, rhoVec.deriv_wrt_params())  # may generate overflow, but OK
    #    _fas(d2pr_dErhos, [0, EVec.gpindices, self.model.prep(rholabel).gpindices],
    #         _np.dot(_np.transpose(EVec.deriv_wrt_params()), derivWrtAnyEvec))
    #
    #    #Note: these 2nd derivatives are non-zero when the spam vectors have
    #    # a more than linear dependence on their parameters.
    #    if self.model.prep(rholabel).has_nonzero_hessian():
    #        derivWrtAnyRhovec = scale * _np.dot(E, prod)  # may overflow, but OK
    #        d2pr_d2rhos = _np.zeros((1, self.model.num_params, self.model.num_params))
    #        _fas(d2pr_d2rhos, [0, self.model.prep(rholabel).gpindices, self.model.prep(rholabel).gpindices],
    #             _np.tensordot(derivWrtAnyRhovec, self.model.prep(rholabel).hessian_wrt_params(), (1, 0)))
    #        # _np.einsum('ij,jkl->ikl', derivWrtAnyRhovec, self.model.prep(rholabel).hessian_wrt_params())
    #    else:
    #        d2pr_d2rhos = 0
    #
    #    if self.model.effect(elabel).has_nonzero_hessian():
    #        derivWrtAnyEvec = scale * _np.transpose(_np.dot(prod, rho))  # may overflow, but OK
    #        d2pr_d2Es = _np.zeros((1, self.model.num_params, self.model.num_params))
    #        _fas(d2pr_d2Es, [0, self.model.effect(elabel).gpindices, self.model.effect(elabel).gpindices],
    #             _np.tensordot(derivWrtAnyEvec, self.model.effect(elabel).hessian_wrt_params(), (1, 0)))
    #        # _np.einsum('ij,jkl->ikl',derivWrtAnyEvec,self.model.effect(elabel).hessian_wrt_params())
    #    else:
    #        d2pr_d2Es = 0
    #
    #    ret = d2pr_dErhos + _np.transpose(d2pr_dErhos, (0, 2, 1)) + \
    #        d2pr_drhos + _np.transpose(d2pr_drhos, (0, 2, 1)) + \
    #        d2pr_dEs + _np.transpose(d2pr_dEs, (0, 2, 1)) + \
    #        d2pr_d2rhos + d2pr_d2Es + d2pr_dOps2
    #    # Note: add transposes b/c spam terms only compute one triangle of hessian
    #    # Note: d2pr_d2rhos and d2pr_d2Es terms are always zero
    #
    #    _np.seterr(**old_err)
    #
    #    if return_deriv:
    #        if return_pr: return ret, dpr, p
    #        else: return ret, dpr
    #    else:
    #        if return_pr: return ret, p
    #        else: return ret
    #
    #def _check(self, eval_tree, pr_mx_to_fill=None, d_pr_mx_to_fill=None, h_pr_mx_to_fill=None, clip_to=None):
    #    # compare with older slower version that should do the same thing (for debugging)
    #    master_circuit_list = eval_tree.compute_circuits(permute=False)  # raw circuits
    #
    #    for spamTuple, (fInds, gInds) in eval_tree.spamtuple_indices.items():
    #        circuit_list = master_circuit_list[gInds]
    #
    #        if pr_mx_to_fill is not None:
    #            check_vp = _np.array([self._prs(spamTuple[0], [spamTuple[1]], circuit, clip_to, False)[0]
    #                                  for circuit in circuit_list])
    #            if _nla.norm(pr_mx_to_fill[fInds] - check_vp) > 1e-6:
    #                _warnings.warn("norm(vp-check_vp) = %g - %g = %g" %
    #                               (_nla.norm(pr_mx_to_fill[fInds]),
    #                                _nla.norm(check_vp),
    #                                _nla.norm(pr_mx_to_fill[fInds] - check_vp)))  # pragma: no cover
    #
    #        if d_pr_mx_to_fill is not None:
    #            check_vdp = _np.concatenate(
    #                [self._dpr(spamTuple, circuit, False, clip_to)
    #                 for circuit in circuit_list], axis=0)
    #            if _nla.norm(d_pr_mx_to_fill[fInds] - check_vdp) > 1e-6:
    #                _warnings.warn("norm(vdp-check_vdp) = %g - %g = %g" %
    #                               (_nla.norm(d_pr_mx_to_fill[fInds]),
    #                                _nla.norm(check_vdp),
    #                                _nla.norm(d_pr_mx_to_fill[fInds] - check_vdp)))  # pragma: no cover
    #
    #        if h_pr_mx_to_fill is not None:
    #            check_vhp = _np.concatenate(
    #                [self._hpr(spamTuple, circuit, False, False, clip_to)
    #                 for circuit in circuit_list], axis=0)
    #            if _nla.norm(h_pr_mx_to_fill[fInds][0] - check_vhp[0]) > 1e-6:
    #                _warnings.warn("norm(vhp-check_vhp) = %g - %g = %g" %
    #                               (_nla.norm(h_pr_mx_to_fill[fInds]),
    #                                _nla.norm(check_vhp),
    #                                _nla.norm(h_pr_mx_to_fill[fInds] - check_vhp)))  # pragma: no cover



class CPTPMatrixForwardSimTester(MatrixForwardSimTester):
    @classmethod
    def setUpClass(cls):
        super(CPTPMatrixForwardSimTester, cls).setUpClass()
        cls.model = cls.model.copy()
        cls.model.set_all_parameterizations("CPTP")  # so gates have nonzero hessians


class MapForwardSimTester(ForwardSimBase, BaseCase):
    @classmethod
    def setUpClass(cls):
        super(MapForwardSimTester, cls).setUpClass()
        cls.model = cls.model.copy()
        cls.model.sim = MapForwardSimulator()
