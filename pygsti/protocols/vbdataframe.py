"""
Techniques for manipulating benchmarking data stored in a Pandas DataFrame.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
import pandas as _pandas
import numpy as _np
from scipy.stats import chi2 as _chi2


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


def polarization_to_success_probability(p, na):
    """
    Inverse of success_probability_to_polarization.
    """
    return p * (1 - 1 / 2**n) + 1 / 2**n


def success_probability_to_polarization(s, n):
    """
    Maps a success probablity s for an n-qubit circuit to
    the polarization, defined by `p = (s - 1/2^n)/(1 - 1/2^n)`
    """
    return (s - 1 / 2**n) / (1 - 1 / 2**n)


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
    def __init__(self, df, x_axis='Depth', y_axis='Width', x_values=None, y_values=None,
                 edesign=None):
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
        assert(metric in ('polarization', 'success_probabilities'))

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
