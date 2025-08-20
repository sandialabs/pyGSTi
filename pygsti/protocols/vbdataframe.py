"""
Techniques for manipulating benchmarking data stored in a Pandas DataFrame.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
from __future__ import annotations
from typing import Callable, Literal, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    try:
        import matplotlib as _mp
    except:
        pass


import pandas as _pandas
import numpy as _np
import tqdm as _tqdm
from scipy.stats import chi2 as _chi2

from pygsti.protocols import (
    ExperimentDesign as _ExperimentDesign,
    FreeformDesign as _FreeformDesign,
    ProtocolData as _ProtocolData
)

from pygsti.tools.mcfetools import (hamming_distance_counts,
                                    effective_polarization,
                                    predicted_process_fidelity_for_central_pauli_mcs,
                                    fidelity_to_polarization,
                                    polarization_to_success_probability,
                                    rc_predicted_process_fidelity,
                                    rc_bootstrap_predicted_pfid
                                    )

def _calculate_summary_statistic(x, statistic, lower_cutoff=None):
    """
    Utility function that returns statistic(x), or the maximum
    of statistic(x) and lower_cutoff if lower_cutoff is not None.
    """
    if len(x) == 0 or _np.all(_np.isnan(x)): return _np.nan
    if statistic == 'mean': func = _np.nanmean
    elif statistic == 'max' or statistic == 'monotonic_max': func = _np.nanmax
    elif statistic == 'min' or statistic == 'monotonic_min': func = _np.nanmin
    elif statistic == 'min_w_nan': func = _np.min
    else:
        raise ValueError("{} is an unknown statistic!".format(statistic))
    v = func(x)
    if lower_cutoff is None:
        return v
    else:
        return _np.max([v, lower_cutoff])


def classify_circuit_shape(success_probabilities, total_counts, threshold, significance=0.05):

    """
    Utility function for computing "capability regions", as introduced in "Measuring the
    Capabilities of Quantum Computers" arXiv:2008.11294.

    Returns an integer that classifies the input list of success probabilities (SPs) as
    either

    * "success": all SPs above the specified threshold, specified by the int 2.
    * "indeterminate": some SPs are above and some are below the threshold, specified by the int 1.
    * "fail": all SPs are below the threshold, specified by the int 0.

    This classification is based on a hypothesis test whereby the null hypothesis is "success"
    or "fail". That is, the set of success probabilities are designated to be "indeterminate"
    only if there is statistically significant evidence that at least one success probabilities
    is above the threshold, and at least one is below. The details of this hypothesis testing
    are given in the Section 8.B.5 in the Supplement of arXiv:2008.11294.

    Parameters
    ----------
    success_probabilities : list
        List of success probabilities, should all be in [0,1].

    total_counts : list
        The number of samples from which the success probabilities where computed.

    threshold : float
        The threshold for designating a success probability as "success".

    significance : float, optional
        The statistical significance for the hypothesis test.

    Returns
    -------
    int in (2, 1, 0), corresponding to ("success", "indeterminate", "fail") classifications.
    If the SPs list is length 0 then NaN is returned, and if it contains only NaN elements then
    0 is returned. Otherwise, all NaN elements are ignored.
    """
    # If there is no data we return NaN
    if len(success_probabilities) == 0:
        return _np.nan  # No data.
    # NaN data is typically used to correspond to execution failure. If all of the circuits
    # fail to execute we classify this as "fail" (set here), but otherwise we ignore any
    # NaN instances.
    if all([_np.isnan(s) for s in success_probabilities]):
        return 0

    def pVal(p, total_counts, threshold, direction):
        """
        The p-value for a log-likelihood ratio test for whether p is above or below
        the given threshold.
        """
        p = _np.max([p, 1e-10])  # avoid log(0)
        if direction == 'above' and p >= threshold: return 1
        if direction == 'below' and p <= threshold: return 1
        s = p * total_counts
        llr = -2 * s * (_np.log(threshold) - _np.log(p))
        llr += -2 * (total_counts - s) * (_np.log(1 - threshold) - _np.log(1 - p))
        return 1 - _chi2.cdf(llr, 1)

    # Calculates the p values. This ignores nan values in the success probabilitis list.
    pvalsAbove = [pVal(p, c, threshold, 'above') for p, c in zip(success_probabilities, total_counts) if c > 0]
    pvalsBelow = [pVal(p, c, threshold, 'below') for p, c in zip(success_probabilities, total_counts) if c > 0]

    # Implements the hypothesis test (this is a Benjamini-Hochberg test).
    pvalsAbove.sort()
    pvalsBelow.sort()
    m = len(pvalsAbove)
    test_above = [pval < significance * (k + 1) / m for k, pval in enumerate(pvalsAbove)]
    test_below = [pval < significance * (k + 1) / m for k, pval in enumerate(pvalsBelow)]
    reject_all_above = any(test_above)
    reject_all_below = any(test_below)

    # If the hypothesis test doesn't reject the hypothesis that they're all above the threshold, and it does
    # reject the hypothesis that they're all below the threshold we say that all the circuits
    # suceeded.
    if reject_all_below and (not reject_all_above):
        return 2
    # If the hypothesis test doesn't reject the hypothesis that they're all below the threshold, and it does
    # reject the hypothesis that they're all above the threshold we say that all the circuits
    # failed.
    elif (not reject_all_below) and reject_all_above:
        return 0
    # If we're certain that there's at least one circuit above the threshold and at least one below, we
    # say that this is an "indeterminate" shape.
    elif reject_all_above and reject_all_below:
        return 1
    # If we don't reject either null hypothesis we don't assign "indeterminate", but we need to decide
    # whether to classify the list of success probabilities as all above or below threshold. We designate
    # the success probabilities as "success"  if the maximum SP is further above the threshold than
    # the minimum SP is below (and vice versa).
    elif (not reject_all_below) and (not reject_all_above):
        below_score = threshold - _np.nanmin(success_probabilities)
        above_score = _np.nanmax(success_probabilities) - threshold
        if above_score > below_score:
            return 2
        else:
            return 0


class VBDataFrame(object):
    """
    A class for storing a DataFrame that contains volumetric benchmarking data, and that
    has methods for manipulating that data and creating volumetric-benchmarking-like plots.
    """
    def __init__(self,
                 df: _pandas.DataFrame,
                 x_axis: str = 'Depth',
                 y_axis: str = 'Width',
                 x_values: Optional[str] = None,
                 y_values: Optional[str] = None,
                 edesign: Optional[_ExperimentDesign] = None):
        """
        Initialize a VBDataFrame object.

        Parameters
        ----------
        df : Pandas DataFrame
            A DataFrame that contains the volumetric benchmarking data. This sort of
            DataFrame can be created using ByBepthSummaryStatics protocols and the
            to_dataframe() method of the created results object.

        x_axis : string, optional
            A VBDataFrame is intended to create volumetric-benchmarking-like plots
            where performance is plotted on an (x, y) grid. This specifies what the
            x-axis of these plots should be. It should be a column label in the DataFrame.

        y_axis : string, optional
            A VBDataFrame is intended to create volumetric-benchmarking-like plots
            where performance is plotted on an (x, y) grid. This specifies what the
            y-axis of these plots should be. It should be a column label in the DataFrame.

        x_values : string or None, optional

        x_values : string or None, optional

        edesign : ExperimentDesign or None, optional
            The ExperimentDesign that corresponds to the data in the dataframe. This
            is not currently used by any methods in the VBDataFrame.
        """
        self.dataframe = df
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.edesign = edesign

        if x_values is None:
            self.x_values = list(set(df[x_axis]))
            self.x_values.sort()
        else:
            self.x_values = x_values
        if y_values is None:
            self.y_values = list(set(df[y_axis]))
            self.y_values.sort()
        else:
            self.y_values = y_values

    @classmethod
    def from_mirror_experiment(cls,
                               unmirrored_design: _FreeformDesign,
                               mirrored_data: _ProtocolData,
                               include_dropped_gates: bool = False,
                               bootstrap: bool = True,
                               num_bootstraps: int = 50,
                               rand_state: Optional[_np.random.RandomState] = None,
                               verbose: bool = False,
                               ) -> VBDataFrame:
        """
        Create a dataframe from MCFE data and edesigns.

        Parameters
        ----------
        unmirrored_design: pygsti.protocols.protocol.FreeformDesign
            Edesign containing the circuits whose process fidelities are to be estimated.

        mirrored_data: pygsti.protocols.protocol.ProtocolData
            Data object containing the full mirror edesign and the outcome counts for each
            circuit in the full mirror edesign.

        include_dropped_gates: bool
            Whether to include the number of gates dropped from each subcircuit during
            subcircuit creation. This flag should be set to False for noise benchmark and
            fullstack benchmark analysis, but can be set to True for subcircuit benchmark
            analysis. Default is False.

        bootstrap: bool
            Toggle the calculation of error bars from bootstrapped process fidelity calculations. If True,
            error bars are calculated. If False, error bars are not calculated.

        num_bootstraps: int
            Number of samples to draw from the bootstrapped process fidelity calculations. This argument
            is ignored if 'bootstrap' is False.

        rand_state: np.random.RandomState
            random state used to seed bootstrapping. If 'bootstrap' is set to False, this argument is ignored.

        verbose: bool
            Toggle print statements with debug information. If True, print statements are
            turned on. If False, print statements are omitted.

        Returns
        ---------
        VBDataFrame
            A VBDataFrame whose dataframe contains calculated MCFE values and circuit statistics.
        
        """

        eff_pols = {k: {} for k in mirrored_data.edesign.keys()}
    
        # Stats about the base circuit
        dropped_gates = {}
        occurrences = {}

        # Get a dict going from id to circuit for easy lookup
        reverse_circ_ids = {}
        for k,v in unmirrored_design.aux_info.items():
            if isinstance(v, dict):
                reverse_circ_ids[v['id']] = k
            else:
                for entry in v:
                    reverse_circ_ids[entry['id']] = k
        seen_keys = set()

        for c in _tqdm.tqdm(mirrored_data.dataset, ascii=True, desc='Calculating effective polarizations'):
            for edkey, ed in mirrored_data.edesign.items():
                auxlist = ed.aux_info.get(c, None)
                if auxlist is None:
                    continue
                elif isinstance(auxlist, dict):
                    auxlist = [auxlist]

                for aux in auxlist:
                    if edkey.endswith('ref'):
                        # For reference circuits, only width matters, so aggregate on that now
                        key = (aux['width'], c.line_labels)
                    else:
                        key = (aux['base_aux']['width'], aux['base_aux']['depth'], aux['base_aux']['id'], c.line_labels)


                    if edkey == 'br':
                        base_line_labels = reverse_circ_ids[aux['base_aux']['id']].line_labels
                        if c.line_labels != base_line_labels:
                            raise RuntimeError('line labels permuted')

                    # Calculate effective polarization
                    hdc = hamming_distance_counts(mirrored_data.dataset[c], c, (aux['idealout'],))
                    ep = effective_polarization(hdc)
                    
                    # Append to other mirror circuit samples
                    eps = eff_pols[edkey].get(key, [])
                    eps.append(ep)
                    eff_pols[edkey][key] = eps

                    if edkey.endswith('ref') or key in seen_keys:
                        # Skip statistic gathering for reference circuits or base circuits we've seen already
                        continue

                    # TODO: add 2Q gate density metric

                    if include_dropped_gates:
                        dropped_gates[key] = aux['base_aux']['dropped_gates']
                    occurrences[key] = len(auxlist)

                    seen_keys.add(key)

        # Calculate process fidelities
        df_data = {}
        for i, key in enumerate(sorted(list(seen_keys), key=lambda x: x[2])):
            cp_pfid = cp_pol = cp_success_prob = None
            if 'cp' in mirrored_data.edesign and 'cpref' in mirrored_data.edesign:
                if verbose and i == 0: print('Central pauli data detected, computing CP process fidelity')
                cp_pfid = predicted_process_fidelity_for_central_pauli_mcs(eff_pols['cp'][key],
                                                                           eff_pols['cpref'][(key[0], key[3])],
                                                                           key[0])
                
                cp_pol = fidelity_to_polarization(cp_pfid, key[0])
                cp_success_prob = polarization_to_success_probability(cp_pol, key[0])
                
            rc_pfid = rc_pol = rc_success_prob = rc_pfid_stdev = None
            if 'rr' in mirrored_data.edesign and 'br' in mirrored_data.edesign and 'ref' in mirrored_data.edesign:
                if verbose and i == 0: print('Random compilation data detected, computing RC process fidelity')

            
                if verbose:
                    spam_ref_key = (key[0], key[3])
                    print(f'key for SPAM reference circuits: {spam_ref_key}')
                    print(f"number of SPAM reference circuits: {len(eff_pols['ref'][spam_ref_key])}")

                rc_pfid = rc_predicted_process_fidelity(eff_pols['br'][key],
                                                     eff_pols['rr'][key],
                                                     eff_pols['ref'][(key[0], key[3])],
                                                     key[0])
                
                rc_pol = fidelity_to_polarization(rc_pfid, key[0]) # should this be fidelity_to_polarization?
                rc_success_prob = polarization_to_success_probability(rc_pol, key[0])

                if bootstrap:
                # do bootstrapping to obtain confidence intervals
                    rc_pfid_stdev = rc_bootstrap_predicted_pfid(brs=eff_pols['br'][key],
                                                            rrs=eff_pols['rr'][key],
                                                            refs=eff_pols['ref'][(key[0], key[3])],
                                                            n=key[0],
                                                            num_bootstraps=num_bootstraps,
                                                            rand_state=rand_state
                                                            )

            data_dict = {'Width': key[0], 'Depth': key[1], 'Circuit Id': key[2],
                        'CP Process Fidelity': cp_pfid,
                        'CP Polarization': cp_pol, 'CP Success Probability': cp_success_prob,
                        'RC Process Fidelity': rc_pfid, 'RC Process Fidelity stdev': rc_pfid_stdev,
                        'RC Polarization': rc_pol, 'RC Success Probability': rc_success_prob,
                        'Occurrences': occurrences[key],
                        }
            
            if include_dropped_gates:
                data_dict['Dropped Gates'] = dropped_gates[key]
                
        # Depth is doubled for conventions (same happens for RMC/PMC)
            df_data[i] = data_dict
            

        df = _pandas.DataFrame.from_dict(df_data, orient='index')
        df = df.sort_values(by='Circuit Id')
        df = df.reset_index(drop=True)

        return cls(df, x_axis='Depth', y_axis='Width')
        

    def select_column_value(self, column_label, column_value):
        """
        Filters the dataframe, by discarding all rows of the
        dataframe for which the column labelled `column_label`
        does not have `column_value`.

        Parameters
        ----------
        column_label : string
            The label of the column whose value is to be filtered on.

        column_value : varied
            The value of the column.

        Returns
        -------
        VBDataFrame
            A new VBDataFrame that has had the filtering applied
            to its dataframe.
        """
        df = self.dataframe[self.dataframe[column_label] == column_value]
        return VBDataFrame(df, self.x_axis, self.y_axis, self.x_values.copy(), self.y_values.copy(), self.edesign)

    def filter_data(self, column_label, metric='polarization', statistic='mean', indep_x=True, threshold=1 / _np.e,
                    verbosity=0):
        """
        Filters the dataframe, by selecting the "best" value at each (x, y) (typically corresponding to circuit
        shape) for the column specified by `column_label`. Returns a VBDataFrame whose data that contains only
        one value for the column labelled by `column_label` for each (x, y).

        Parameters
        ----------
        column_label : string
            The label of the column whose "best" value at each circuit shape is to be selected. For example,
            this could be "Qubits", to select only the data for the best qubit subset at each circuit shape.

        metric : string, optional
            The data to be used as the figure-of-merit for performance at each (x, y). Must be a column
            of the dataframe.

        statistics : string, optional
            The statistic to apply to the data specified by `metric` the data at (x, y)
            into a scalar. Allowed values are:
            - 'max'
            - 'min'
            - 'mean'

        indep_x : bool, optional
            If True, then an independent value, for the column, is selected at each (x, y) value. If
            False, then the same value for the column is selected for every x value for a given y.

        threshold : float, optional.
            Does nothing if indep_x is True. If indep_x is False, then 'metric' and 'statistic' are
            not enough to uniquely decide which column value is best. In this case, the value is chosen
            that, for each y in (x,y), maximizes the x value at which the figure-of-merit (as specified
            by the metric and statistic) drops below the threshold. If there are multiple values that
            drop below the threshold at the same x (or the figure-of-merit never drops below the threshold
            for multiple values), then value with the larger figure-of-merit at that x is chosen.

        Returns
        -------
        VBDataFrame
            A new VBDataFrame that has had the filtering applied to its dataframe.
        """
        df = self.dataframe.copy()

        if indep_x:
            for x in self.x_values:
                tempdf_x = df[df[self.x_axis] == x]
                for y in self.y_values:
                    tempdf_xy = tempdf_x[tempdf_x[self.y_axis] == y]
                    possible_column_values = list(set(tempdf_xy[column_label]))
                    scores = []
                    if len(possible_column_values) == 0:
                        if verbosity > 0:
                            print('at ({}, {}) there is no data... selected None'.format(x, y))
                    elif len(possible_column_values) == 1:
                        if verbosity > 0:
                            print('at ({}, {}) only one option: {}'.format(x, y, possible_column_values[0]))
                    else:
                        for cv in possible_column_values:
                            scores.append(_calculate_summary_statistic(tempdf_xy[tempdf_xy[column_label] == cv][metric],
                                                                       statistic))
                        column_value_selection = possible_column_values[_np.argmax(scores)]
                        if verbosity > 0:
                            print('at ({}, {}) selected {}'.format(x, y, column_value_selection))
                        # For all rows where (self.x_axis,self.y_axis) value is (x,y), filter out all values where
                        # column_label does not equal column_value_selection
                        df = df[(df[column_label] == column_value_selection) | (df[self.y_axis] != y)
                                | (df[self.x_axis] != x)]

        else:
            for y in self.y_values:
                tempdf_y = df[df[self.y_axis] == y]
                possible_column_values = list(set(tempdf_y[column_label]))
                if verbosity > 0:
                    print("  Selecting from possible Values ", possible_column_values)
                if len(possible_column_values) == 0:
                    if verbosity > 0: print('at {} = {}  there is no data... selected None'.format(self.y_axis, y))
                elif len(possible_column_values) == 1:
                    if verbosity > 0: print('at {} = {} only one option: {}'.format(self.y_axis, y,
                                                                                    possible_column_values[0]))
                else:
                    current_best_boundary_index = -1
                    current_best_selections = []
                    current_best_selections_score = []
                    for cv in possible_column_values:
                        scores_for_cv = [_calculate_summary_statistic(tempdf_y[(tempdf_y[column_label] == cv)
                                         & (tempdf_y[self.x_axis] == x)][metric], statistic)for x in self.x_values]
                        above_threshold = [s > threshold if not _np.isnan(s) else _np.nan for s in scores_for_cv]
                        try:
                            above_threshold = above_threshold[:above_threshold.index(False)]
                        except:
                            pass
                        above_threshold.reverse()
                        try:
                            boundary_index = len(above_threshold) - 1 - above_threshold.index(True)
                            score = scores_for_cv[boundary_index]
                        except:
                            boundary_index = -1
                            score = scores_for_cv[0]
                        if boundary_index > current_best_boundary_index:
                            current_best_selections = []
                            current_best_selections_score = []
                        if boundary_index >= current_best_boundary_index:
                            current_best_selections.append(cv)
                            current_best_selections_score.append(score)
                            current_best_boundary_index = boundary_index

                    best_selection_index = _np.argmax(current_best_selections_score)
                    column_value_selection = current_best_selections[best_selection_index]

                    if verbosity > 0:
                        print('at {} = {}, selected {}'.format(self.y_axis, y, column_value_selection))
                    # For all rows where self.y_axis value is y, filter out all values where column_label does
                    # not equal column_value_selection
                    df = df[(df[column_label] == column_value_selection) | (df[self.y_axis] != y)]

        return VBDataFrame(df, self.x_axis, self.y_axis, self.x_values.copy(), self.y_values.copy(), self.edesign)

    def vb_data(self, metric='polarization', statistic='mean', lower_cutoff=0., no_data_action='discard'):
        """
        Converts the data into a dictionary, for plotting in a volumetric benchmarking plot. For each
        (x, y) value (as specified by the axes of this VBDataFrame, and typically circuit shape),
        pools all of the data specified by `metric` with that (x, y) and computes the statistic on
        that data defined by `statistic`.

        Parameters
        ----------
        metric : string, optional
            The type of data. Must be a column of the dataframe.

        statistics : string, optional
            The statistic on the data to be computed at each value of (x, y). Options are:

            * 'max': the maximum
            * 'min': the minimum.
            * 'mean': the mean.
            * 'monotonic_max': the maximum of all the data with (x, y) values that are that large or larger
            * 'monotonic_min': the minimum of all the data with (x, y) values that are that small or smaller

            All these options ignore nan values.

        lower_cutoff : float, optional
            The value to cutoff the statistic at: takes the maximum of the calculated static and this value.

        no_data_action: string, optional
            Sets what to do when there is no data, or only NaN data, at an (x, y) value:
            
            * If 'discard' then when there is no data, or only NaN data, for an (x,y) value then this (x,y)
              value will not be a key in the returned dictionary
            * If 'nan' then when there is no data, or only NaN data, for an (x,y) value then this (x,y)
              value will be a key in the returned dictionary and its value will be NaN.
            * If 'min' then when there is no data, or only NaN data, for an (x,y) value then this (x,y)
              value will be a key in the returned dictionary and its value will be the minimal value
              allowed for this statistic, as specified by `lower_cutoff`.

        Returns
        -------
        dict
            A dictionary where the keys are (x,y) tuples (typically circuit shapes) and the values are the
            VB data at that (x, y).
        """
        vb = {}
        assert(no_data_action in ('discard', 'nan', 'min')), "`no_data_action` must be `discard`, `nan` or `min`!"
        df = self.dataframe
        for x in self.x_values:
            for y in self.y_values:
                if statistic == 'monotonic_min':
                    tempdf_xy = df[(df[self.x_axis] <= x) & (df[self.y_axis] <= y)]
                elif statistic == 'monotonic_max':
                    tempdf_xy = df[(df[self.x_axis] >= x) & (df[self.y_axis] >= y)]
                else:
                    tempdf_xy = df[(df[self.x_axis] == x) & (df[self.y_axis] == y)]
                if all(_np.isnan(tempdf_xy[metric])) or len(tempdf_xy[metric]) == 0:
                    if no_data_action == 'discard':
                        pass
                    elif no_data_action == 'min':
                        vb[x, y] = lower_cutoff
                    elif no_data_action == 'nan':
                        vb[x, y] = _np.nan
                else:
                    vb[x, y] = _calculate_summary_statistic(tempdf_xy[metric], statistic, lower_cutoff=lower_cutoff)

        return vb

    def capability_regions(self, metric='polarization', threshold=1 / _np.e, significance=0.05, monotonic=True,
                           nan_data_action='discard'):
        """
        Computes a "capability region" from the data,  as introduced in "Measuring the Capabilities of Quantum
        Computers" arXiv:2008.11294. Classifies each (x,y) value (as specified by the x and y axes of
        the VBDataFrame, which are typically width and depth) as either "success" (the int 2), "indeterminate"
        (the int 1), "fail" (the int 0), or "no data" (NaN).

        Parameters
        ----------
        metric : string, optional
            The type of data. Must be 'polarization' or 'success_probability', and this must be a column
            in the dataframe.

        threshold : float, optional
            The threshold for  "success".

        significance : float, optional
            The statistical significance for the hypothesis tests that are used to classify each circuit
            shape.

        monotonic : bool, optional
            If True, makes the region monotonic, i,e, if (x',y') > (x,y) then the classification for
            (x',y') is less/worse than for (x,y).

        no_data_action : string, optional
            If 'discard' then when there is no data, for an (x,y) value then this (x,y)
            value will not be a key in the returned dictionary. Otherwise the value will be NaN.

        Returns
        -------
        dict
            A dictionary where the keys are (x,y) tuples (typically circuit shapes) and the values are in
            (2, 1, 0, NaN).
        """
        capreg = {}
        assert(metric in ('polarization', 'success_probability'))

        for x in self.x_values:
            tempdf_x = self.dataframe[self.dataframe[self.x_axis] == x]
            for y in self.y_values:
                tempdf_xy = tempdf_x[tempdf_x[self.y_axis] == y]
                if metric == 'polarization':
                    assert(len(set(tempdf_xy['Width'])) <= 1), 'There are circuits with different widths at this' \
                        + '(x,y) value so cannot rescale polarization threshold to a success probability threshold'
                    if len(set(tempdf_xy['Width'])) == 1:
                        w = list(tempdf_xy['Width'])[0]
                        sp_threshold = polarization_to_success_probability(threshold, w)
                    else:
                        sp_threshold = 0  # Doesn't matter what the threshold is as there's no data.
                else:
                    sp_threshold = threshold

                capreg[x, y] = classify_circuit_shape(tempdf_xy['success_probabilities'], tempdf_xy['total_counts'],
                                                      sp_threshold, significance)

        if monotonic:
            for x in self.x_values:
                for i, y in enumerate(self.y_values[1:]):
                    if capreg[x, y] > capreg[x, self.y_values[i]]:
                        capreg[x, y] = capreg[x, self.y_values[i]]
            for y in self.y_values:
                for i, x in enumerate(self.x_values[1:]):
                    if capreg[x, y] > capreg[self.x_values[i], y]:
                        capreg[x, y] = capreg[self.x_values[i], y]

        if nan_data_action == 'discard':
            trimmed_capreg = {}
            for key, val in capreg.items():
                if not _np.isnan(val):
                    trimmed_capreg[key] = val
            capreg = trimmed_capreg

        return capreg
    


    def create_vb_plot(self,
                       title: str,
                       accumulator: Callable = _np.mean,
                       cp_or_rc: Literal['cp', 'rc'] = 'rc',
                       show_dropped_gates: bool = False,
                       dg_accumulator: Callable = _np.mean,
                       cmap: _mp.colors.Colormap = None,
                       margin: float = 0.15,
                       save_fig: bool = False,
                       fig_path: str = None,
                       fig_format: str = None):

        """
        Generate process fidelity volumetric benchmarking (VB) plot from dataframe.
        This function is designed with subcircuit volumetric benchmarking in mind,
        where the x-axis is the subicrcuit depth and the y-axis is the subcircuit
        width.

        Parameters
        ------------
        title : str
            The title of the plot.

        accumulator : callable, optional
            Function used to accumulate process fidelities for a (width, depth) pair
            on the VB plot. Default is np.mean.

        cp_or_rc : str, optional
            Whether the process fidelities were computed via randomly compiled
            circuits ('rc') or central Pauli mirroring ('cp'). Default is 'rc'.

        show_dropped_gates : bool, optional
            Whether the plot should visualize the average (see dg_accumulator)
            number of dropped gates for each subcircuit width-depth pair. 
            Subcircuit sampling can drop gates when a gate has only partial support
            on the qubits in the selected width subset.

        dg_accumulator : callable, optional
            Function used to accumulate the dropped gate counts
            for a (width, depth) pair on the VB plot. Default is np.mean.

        cmap : matplotlib.colors.Colormap, optional
            Colormap to use for plotting process fidelities. Default is spectral.

        margin : float, optional
            Margin between adjacent width-depth pairs in the VB plot.
            Default is 0.15.

        save_fig : bool, optional
            Whether to write the generated VB plot to file. Default is False.

        fig_path : str, optional
            If `save_fig` is set to True, this argument is used as the path
            the figure is saved to. If `save_fig` is False, this argument
            is ignored. Default is None

        fig_format : str, optional
            If `save_fig` is set to True, this argument is the file format
            for the generated VB plot. If `save_fig` is False, this argument
            is ignored. Acceptable values are any file format recognized by matplotlib.
        """
        try:
            import matplotlib.pyplot as _plt
            import matplotlib as _mp
        except:
            raise RuntimeError('matplotlib is required for dataframe plotting, and does not appear to be installed.')
        
        if cmap is None:
            cmap = _mp.colormaps['Spectral']


        fig = _plt.figure(figsize=(5.5,8))
        ax = _plt.gca()
        ax.set_aspect('equal')

        x_axis = 'Depth'
        y_axis = 'Width'
        
        if cp_or_rc == 'rc':
            c_axis = 'RC Process Fidelity'
        elif cp_or_rc == 'cp':
            c_axis = 'CP Process Fidelity'
        else:
            raise ValueError(f"invalid argument passed to 'cp_or_rc': {cp_or_rc}")
        
        xticks = self.dataframe[x_axis].unique()
        yticks = self.dataframe[y_axis].unique()
        
        xmap = {j: i for i, j in enumerate(xticks)}
        ymap = {j: i for i, j in enumerate(yticks)}
        
        acc_values = self.dataframe.groupby([x_axis, y_axis])[c_axis].apply(accumulator)
        if show_dropped_gates:
            dg_values = self.dataframe.groupby([x_axis, y_axis])["Dropped Gates"].apply(dg_accumulator)

        for xlbl, ylbl in acc_values.keys():
            x = xmap[xlbl]
            y = ymap[ylbl]

            side = 0.5-margin
            
            cval = acc_values[(xlbl, ylbl)]
            if show_dropped_gates:
                dgval = dg_values[(xlbl, ylbl)]
                # Inner square: Smaller by ratio of dropped gates
                inside = side * (xlbl-dgval)/xlbl

            else:
                inside = side
            
            inpoints = [(x-side, y-side), (x-side, y+inside), (x+inside , y+inside), (x+inside, y-side)]
            tin = _plt.Polygon(inpoints, color=cmap(cval))
            ax.add_patch(tin)
            
            # Outer square
            outer_points = [(x-side, y-side), (x-side, y+side), (x+side, y+side), (x+side, y-side)]
            tout = _plt.Polygon(outer_points, edgecolor='k', fill=None)
            ax.add_patch(tout)
        
        _plt.xlabel(x_axis, {'size': 20})
        _plt.ylabel(y_axis, {'size': 20})
        _plt.xticks(list(range(len(xticks))), labels=xticks)
        _plt.yticks(list(range(len(yticks))), labels=yticks)
        _plt.xlim([-0.5, len(xticks)-0.5])
        _plt.ylim([-0.5, len(yticks)-0.5])
        ax.tick_params(axis='both', which='major', labelsize=14)
        _plt.title(title, {"size": 24})

        if save_fig:
            _plt.savefig(fig_path, format=fig_format)