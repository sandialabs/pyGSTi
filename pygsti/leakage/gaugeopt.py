#***************************************************************************************************
# Copyright 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from __future__ import annotations

import copy
from typing import Dict, List, Optional, TYPE_CHECKING, Union

from pygsti.baseobjs.basis import Basis

if TYPE_CHECKING:
    from pygsti.models import ExplicitOpModel
    from pygsti.protocols.gst import GSTGaugeOptSuite, ModelEstimateResults


def _direct_sum_unitary_group(subspace_bases, full_basis, triviality_flags=None,
                              level_partition=None):
    """
    Build a gauge group that acts as an independent unitary on each summand of a
    direct-sum decomposition H = H₀ ⨁ H₁ ⨁ ... of Hilbert space.

    `full_basis` is a Hilbert--Schmidt basis for M[H], and `subspace_bases[i]` is a basis
    for M[Hᵢ]. Each summand contributes its own gauge subgroup, chosen by its
    Hilbert--Schmidt dimension ``sb.dim``:

      * TrivialGaugeGroup, if the summand is flagged trivial -- by default whenever
        ``sb.dim == 1``, i.e. Hᵢ is a single level with no unitary freedom to fix;
      * UnitaryGaugeGroup, if ``sb.dim > 1``; and
      * U1Group otherwise (reachable only when `triviality_flags` explicitly marks a
        one-dimensional summand as non-trivial).

    The subgroups are combined into a DirectSumUnitaryGroup on `full_basis`.
    `triviality_flags`, if given, overrides the default per-summand choice and must have
    the same length as `subspace_bases`. `level_partition`, if given, lists the
    standard-basis levels of H that each summand occupies (one sub-list per summand),
    allowing interleaved rather than contiguous summands; see
    `pygsti.models.gaugegroup.DirectSumUnitaryGroup`.
    """
    from pygsti.models.gaugegroup import (UnitaryGaugeGroup,
        DirectSumUnitaryGroup, U1Group, TrivialGaugeGroup)

    if triviality_flags is None:
        triviality_flags = [ sb.dim == 1 for sb in subspace_bases ]
    else:
        assert len(triviality_flags) == len(subspace_bases)

    subgroups : list[Union[TrivialGaugeGroup, U1Group, UnitaryGaugeGroup]] = []
    for sb, tf in zip(subspace_bases, triviality_flags):
        if tf:
            g = TrivialGaugeGroup(sb.state_space)
        elif sb.dim > 1:
            g = UnitaryGaugeGroup(sb.state_space, sb)
        else:
            g = U1Group()
        subgroups.append(g)
    g_full = DirectSumUnitaryGroup(tuple(subgroups), full_basis, level_partition=level_partition)
    return g_full


def _leakage_direct_sum_group(basis: Basis):
    """
    Build the U(k) ⨁ U(m) direct-sum gauge group implied by a leakage basis, where k is
    the dimension of the computational subspace C designated by `basis` and m is the
    dimension of its orthogonal complement (the leakage subspace).

    The subgroup factors are specified by Hilbert--Schmidt bases per
    `_direct_sum_unitary_group`: the default basis name for each factor's dimension
    ('pp' for a qubit-sized factor, 'gm' otherwise). With the default triviality flags,
    a one-dimensional leakage factor contributes a TrivialGaugeGroup.

    DirectSumUnitaryGroup elements act as block-diagonal unitaries, but they accept a
    `level_partition` that lets each block sit on interleaved standard-basis levels. This
    construction therefore supports any leakage basis whose computational effect E is a
    *coordinate* projector -- i.e. E is diagonal with 0/1 entries, so the computational
    subspace is spanned by some subset of standard-basis levels (which need not be the
    leading k levels). Tensor-product leakage bases like pp ⨂ l2p1, whose computational
    levels {0,1,3,4} are interleaved with leakage levels, are handled this way.

    We raise NotImplementedError only if E is a projector onto a genuinely non-coordinate
    subspace (E not diagonal), which does not arise in leakage modeling -- there a
    permutation no longer suffices and a general change of basis would be required.
    """
    import numpy as np
    from pygsti.baseobjs.basis import default_basis_for_udims
    from pygsti.leakage.core import computational_effect

    E = computational_effect(basis)
    udim = E.shape[0]
    k = int(round(np.trace(E).real))
    m = udim - k
    if m == 0:
        raise ValueError(
            f"basis {basis} does not imply leakage modeling (its computational "
            f"subspace is the full {udim}-dimensional Hilbert space), so there is "
            f"no leakage-preserving direct-sum gauge group to build."
        )
    diag = np.diag(E).real
    off_diagonal = E - np.diag(np.diag(E))
    is_coordinate_projector = (
        np.allclose(off_diagonal, 0, atol=1e-10)
        and np.all(np.isclose(diag, 0, atol=1e-10) | np.isclose(diag, 1, atol=1e-10))
    )
    if not is_coordinate_projector:
        raise NotImplementedError(
            f"The computational effect of basis {basis} is a projector onto a "
            f"non-coordinate subspace (it is not diagonal in the standard basis). "
            f"DirectSumUnitaryGroup can place a summand on any subset of standard-basis "
            f"levels via a level permutation, but a computational subspace that mixes "
            f"levels would require a general change of basis, which is not implemented."
        )
    comp_levels = [i for i in range(udim) if diag[i] > 0.5]
    leak_levels = [i for i in range(udim) if diag[i] <= 0.5]
    sub_bases = [
        Basis.cast(default_basis_for_udims([k]), k**2),
        Basis.cast(default_basis_for_udims([m]), m**2),
    ]
    # A contiguous leading computational block is the block_diag default, so leave the
    # partition as None in that (common, qubit-sized) case; only pass an explicit
    # partition when the computational levels are interleaved with leakage levels.
    level_partition = None if comp_levels == list(range(k)) else [comp_levels, leak_levels]
    return _direct_sum_unitary_group(sub_bases, basis, level_partition=level_partition)


def lagoified_gopparams_dicts(gopparams_dicts: List[Dict]) -> List[Dict]:
    """
    goppparams_dicts is a list-of-dicts (LoDs) representation of a gauge optimization suite
    suitable for models without leakage (e.g., a model of a 2-level system).

    This function returns a new gauge optimization suite (also in the LoDs representation)
    by applying leakage-specific modifications to a deep-copy of gopparams_dicts.

    Example
    -------
    Suppose we have a ModelEstimateResults object called `results` that includes a
    CPTPLND estimate, and we want to update the models of that estimate to include
    two types of leakage-aware gauge optimization.

        #
        # Step 1: get the input to this function
        #
        estimate = results.estimates['CPTPLND']
        model    = estimate.models['target']
        stdsuite = GSTGaugeOptSuite(gaugeopt_suite_names=('stdgaugeopt',))
        gopparams_dicts = stdsuite.to_dictionary(model)['stdgaugeopt']
        
        #
        # Step 2: use this function to build our GSTGaugeOptSuite.
        #
        s = lagoified_gopparams_dicts(gopparams_dicts)
        c = lagoified_gopparams_dicts(gopparams_dicts)
        c[-1]['gates_metric'] = 'fidelity'
        c[-1]['spam_metric']  = 'fidelity'
        # ^ this example's "custom" leakage gauge optimization uses an infidelity loss,
        #   rather than the default of squared Frobenius loss.
        specification = {'LAGO-std': s,'LAGO-custom': c}
        gos = GSTGaugeOptSuite(gaugeopt_argument_dicts=specification)
        
        #
        # Step 3: updating `estimate` requires that we modify `results`.
        #
        add_lago_models(results, 'CPTPLND', gos)

    After those lines execute, the `estimate.models` dict will have two new
    key-value pairs, where the keys are 'LAGO-std' and 'LAGO-custom'.
    """
    from pygsti.models.gaugegroup import UnitaryGaugeGroup
    tm = gopparams_dicts[0]['target_model']
    gopparams_dicts = [gp for gp in gopparams_dicts if 'TPSpam' not in str(type(gp.get('_gaugeGroupEl')))]
    # ^ That list will __usually__ be length-1. We use .get since dicts fresh from
    #   GSTGaugeOptSuite.to_dictionary lack the _gaugeGroupEl key (it's filled in at
    #   gauge-optimization time); such dicts are never TPSpam steps, so they're kept.

    gopparams_dicts = copy.deepcopy(gopparams_dicts)
    # ^ Probably a list of length 1.
    for inner_dict in gopparams_dicts:
        inner_dict['method'] = 'L-BFGS-B'
        # ^ We need this optimizer because it doesn't require a gradient oracle.
        #
        inner_dict['leakage_modeling'] = True
        # ^ We use subspace-restricted loss functions that only care about mismatches
        #   between an estimate and a target when restricted to the computational subspace.
        #
        gg = UnitaryGaugeGroup(tm.basis.state_space, tm.basis)
        inner_dict['gauge_group'] = gg
        inner_dict['_gaugeGroupEl'] = gg.compute_element(gg.initial_params)
        # ^ We insist on the unitary gauge group because other common gauge groups
        #   (e.g., FullTPGaugeGroup) impose requirements on tm.basis.
        #
        inner_dict['gates_metric'] = 'frobenius'
        inner_dict['spam_metric']  = 'frobenius'
        # ^ We use Frobenius rather than Frobenius squared to avoid biasing toward
        #   gauges where the leakage is "spread out" across multiple gates.
        #
        inner_dict['item_weights'] = {'gates': 1.0, 'spam': 1.0}
        # ^ The precise weights here are somewhat arbitrary. We place weight on
        #   SPAM because SPAM plays a distinguished role in defining the 
        #   computational subspace.
    
    # Below, we add a final gauge optimization step that preserves separation between 
    # what we've identified as the computational subspace and leakage space. This
    # step uses squared Frobenius norm as an objective, since that's better-behaved
    # than plain Frobenius norm.
    inner_dict = inner_dict.copy()
    gg = _leakage_direct_sum_group(tm.basis)
    inner_dict['gauge_group'] = gg
    inner_dict['_gaugeGroupEl'] = gg.compute_element(gg.initial_params)
    inner_dict['gates_metric'] = 'frobenius squared'
    inner_dict['spam_metric']  = 'frobenius squared'
    inner_dict['item_weights'] = {'gates': 1.0, 'spam': 1.0}
    gopparams_dicts.append(inner_dict)
    return gopparams_dicts


def std_lago_gopsuite(model: ExplicitOpModel) -> dict[str, list[dict]]:
    """
    Return a dictionary of the form {'LAGO': v}, where v is a
    list-of-dicts representation of a gauge optimization suite.

    We construct v by getting the list-of-dicts representation of the
    "stdgaugeopt" suite for `model`, and then changing some of its 
    options to be suitable for leakage-aware gauge optimization. These
    changes are made in the `lagoified_gopparams_dicts` function.
    """
    from pygsti.protocols.gst import GSTGaugeOptSuite
    std_gop_suite = GSTGaugeOptSuite(gaugeopt_suite_names=('stdgaugeopt',))
    std_gos_lods  = std_gop_suite.to_dictionary(model)['stdgaugeopt']  # list of dictionaries
    for d in std_gos_lods:
        d.setdefault('target_model', model)
        # ^ to_dictionary omits target_model (it is normally filled in at gauge-
        #   optimization time), but lagoified_gopparams_dicts needs it to derive
        #   the leakage-aware gauge group from the model's basis.
    lago_gos_lods = lagoified_gopparams_dicts(std_gos_lods)
    gop_params = {'LAGO': lago_gos_lods}
    return gop_params


def add_lago_models(results: ModelEstimateResults, est_key: Optional[str] = None, gos: Optional[GSTGaugeOptSuite] = None, verbosity: int = 0):
    """
    Update each estimate in results.estimates (or just results.estimates[est_key],
    if est_key is not None) with a model obtained by parameterization-preserving
    leakage-aware gauge optimization.
    
    If no gauge optimization suite is provided, then we construct one by making
    appropriate modifications to either the estimate's existing 'stdgaugeopt' suite
    (if that exists) or to the 'stdgaugeopt' suite induced by the target model.
    """
    from pygsti.protocols.gst import GSTGaugeOptSuite, _add_param_preserving_gauge_opt, _add_gauge_opt
    if isinstance(est_key, str):
        if gos is None:
            existing_est  = results.estimates[est_key]
            if 'stdgaugeopt' in existing_est.goparameters:
                std_gos_lods  = existing_est.goparameters['stdgaugeopt']
                lago_gos_lods = lagoified_gopparams_dicts(std_gos_lods)
                gop_params = {'LAGO': lago_gos_lods}
            else:
                gop_params = std_lago_gopsuite(results.estimates[est_key].models['target'])
            gos = GSTGaugeOptSuite(gaugeopt_argument_dicts=gop_params)
        try:
            _add_param_preserving_gauge_opt(results, est_key, gos, verbosity)
        except:
            # parameterization-preserving gauge optimization fails if the seed model
            # uses members other than ComposedState, ComposedOp, ComposedPOVM, etc..
            # So we add a non-preserving gauge optimization here.
            tm = results.estimates[est_key].models['target']
            _add_gauge_opt(results, est_key, gos, tm, tuple())
    elif est_key is None:
        for est_key in results.estimates.keys():
            add_lago_models(results, est_key, gos, verbosity)
    else:
        raise ValueError(
            f"est_key must be a string (a key of results.estimates) or None "
            f"(to update every estimate); got {est_key!r} of type {type(est_key).__name__}."
        )
    return
