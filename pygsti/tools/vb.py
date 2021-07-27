"""
Tools for manipulating benchmarking data stored in Pandas DataFrame.
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
    if len(x) == 0 or _np.all(_np.isnan(x)): return _np.NaN
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


def polarization_to_success_probability(p, n):
    return p * (1 - 1 / 2**n) + 1 / 2**n


def success_probability_to_polarization(s, n):
    return (s - 1 / 2**n) / (1 - 1 / 2**n)


def classify_circuit_shape(success_probabilities, total_counts, threshold, significance=0.05):

    """
    0.5
    -1 : no data
    """
    if len(success_probabilities) == 0: return _np.nan  # No data.
    
    def pVal(p, total_counts, threshold, direction):
        p = _np.max([p, 1e-10])  # avoid log(0)
        if direction == 'above' and p >= threshold: return 1
        if direction == 'below' and p <= threshold: return 1
        s = p * total_counts
        llr = -2 * s * (_np.log(threshold) - _np.log(p))
        llr += -2 * (total_counts - s) * (_np.log(1 - threshold) - _np.log(1 - p))
        return 1 - _chi2.cdf(llr, 1)
    
    # This ignores nan values.
    pvalsAbove = [pVal(p, c, threshold, 'above') for p, c in zip(success_probabilities, total_counts) if c > 0]
    pvalsBelow = [pVal(p, c, threshold, 'below') for p, c in zip(success_probabilities, total_counts) if c > 0]
    
    pvalsAbove.sort()
    pvalsBelow.sort()
    
    # Implement the hypothesis test.
    m = len(pvalsAbove)
    test_above = [pval < significance * (k + 1) / m for k, pval in enumerate(pvalsAbove)]
    test_below = [pval < significance * (k + 1) / m for k, pval in enumerate(pvalsBelow)]
    
    #print(success_probabilities)
    #print(test_above)
    #print(test_below)
    reject_all_above = any(test_above)
    reject_all_below = any(test_below)
    
    # 
    if (not reject_all_below) and (not reject_all_above):
        below_score = threshold - _np.nanmin(success_probabilities)
        above_score = _np.nanmax(success_probabilities) - threshold
        if above_score > below_score:
            return 2
        else:
            return 0
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
    # ...
    else:
        return -1


class VBDataFrame(object):
    
    def __init__(self, df, x_axis='Depth', y_axis='Width', x_values=None, y_values=None,
                 edesign=None):
        
        self.DataFrame = df
        self.FilteredDataFrame = df.copy()
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.edesign = edesign
        self.selected_column_value = {}
        self.filters = {}
        
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

    def reset_filters(self):

        self.FilteredDataFrame = self.DataFrame.copy()
        self.selected_column_value = {}
        self.filters = {}

    def select_column_value(self, column_label, column_value):
        
        #self.selected_column_value[column_label] = column_value
        #self.FilteredDataFrame = self.FilteredDataFrame[self.FilteredDataFrame[column_label] == column_value]
        self.FilteredDataFrame = self.FilteredDataFrame[self.FilteredDataFrame[column_label] == column_value]
    
    def filter_data(self, column_label, column_value=None, metric='polarization', statistic='mean', 
                    indep_x=True, indep_y=True,
                    threshold=1 / _np.e, verbosity=0):

        assert(column_label not in self.selected_column_value.keys()), "Have already filtered on this column value! Use `reset_filters()` first!"
    
        if column_value is not None:
            self.selected_column_value[column_label] = column_value
            self.FilteredDataFrame = self.FilteredDataFrame[self.FilteredDataFrame[column_label] == column_value]
        
        else:

            self.selected_column_value[column_label] = {}
            # mask = _np.array(len(self.FilteredDataFrame), bool)
            if indep_x and indep_y:
                for x in self.x_values:
                    #print(x)
                    tempdf_x = self.FilteredDataFrame[self.FilteredDataFrame[self.x_axis] == x]
                    for y in self.y_values:
                        tempdf_xy = tempdf_x[tempdf_x[self.y_axis] == y]
                        possible_column_values = list(set(tempdf_xy[column_label]))
                        scores = []
                        #if verbosity > 0:
                        #    print("  Selecting from possible Values ", possible_column_values)
                        if len(possible_column_values) == 0:
                            if verbosity > 0:
                                print('at ({}, {}) there is no data... selected None'.format(x, y))
                            self.selected_column_value[column_label][x, y] = None
                        elif len(possible_column_values) == 1:
                            if verbosity > 0:
                                print('at ({}, {}) only one option: {}'.format(x, y, possible_column_values[0]))
                            self.selected_column_value[column_label][x, y] = possible_column_values[0]
                        else:
                            for cv in possible_column_values:
                                scores.append(_calculate_summary_statistic(tempdf_xy[tempdf_xy[column_label] == cv][metric], statistic))
                            column_value_selection = possible_column_values[_np.argmax(scores)]
                            self.selected_column_value[column_label][x, y] = column_value_selection
                            if verbosity > 0:
                                print('at ({}, {}) selected {}'.format(x, y, column_value_selection))
                            #mask = _np.array([q == column_value_selection if xx == x and yy == y else True for q, xx, yy in zip(self.FilteredDataFrame[column_label], self.FilteredDataFrame[self.x_axis], self.FilteredDataFrame[self.y_axis])])
                            #self.FilteredDataFrame = self.FilteredDataFrame[mask]
                            # For all rows where (self.x_axis,self.y_axis) value is (x,y), filter out all values where column_label does not equal column_value_selection
                            self.FilteredDataFrame = self.FilteredDataFrame[(self.FilteredDataFrame[column_label] == column_value_selection) ^ (self.FilteredDataFrame[self.y_axis] != y) ^ (self.FilteredDataFrame[self.x_axis] != x)]
                            
            elif (not indep_x) and indep_y:
                for y in self.y_values:
                    tempdf_y = self.FilteredDataFrame[self.FilteredDataFrame[self.y_axis] == y]
                    possible_column_values = list(set(tempdf_y[column_label]))
                    if verbosity > 0:
                        print("  Selecting from possible Values ", possible_column_values)
                    if len(possible_column_values) == 0:
                        if verbosity > 0: print('at {} = {}  there is no data... selected None'.format(self.y_axis, y))
                        self.selected_column_value[column_label][y] = None
                    elif len(possible_column_values) == 1:
                        if verbosity > 0: print('at {} = {} only one option: {}'.format(self.y_axis, y, possible_column_values[0]))
                        self.selected_column_value[column_label][y] = possible_column_values[0]
                    else:
                        current_best_boundary_index = -1
                        current_best_selections = []
                        current_best_selections_score = []
                        for cv in possible_column_values:
                            scores_for_cv = [_calculate_summary_statistic(tempdf_y[(tempdf_y[column_label] == cv) & (tempdf_y[self.x_axis] == x)][metric], statistic) for x in self.x_values]
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

                        #print(current_best_selections, current_best_selections_score)
                        
                        self.selected_column_value[column_label][y] = column_value_selection
                        if verbosity > 0:
                            print('at {} = {}, selected {}'.format(self.y_axis, y, column_value_selection))
                        #mask = _np.array([q == column_value_selection if yy == y else True for q, xx, yy in zip(, )])
                        # For all rows where self.y_axis value is y, filter out all values where column_label does not equal column_value_selection
                        self.FilteredDataFrame = self.FilteredDataFrame[(self.FilteredDataFrame[column_label] == column_value_selection) ^ (self.FilteredDataFrame[self.y_axis] != y)]
                           
            else:
                raise ValueError("indep_y must be True!")
                
    def get_vb_data(self, metric='polarization', statistic='mean', lower_cutoff=0., no_data_action='nan'):
        """
        Converts the data into a dictionary, for plotting in a volumetric benchmarking plot.

        Parameters
        ----------
        todo

        no_data_action:
            If 'discard' then when there is no data, or only NaN data, for an (x,y) value then this (x,y)
            value will not be a key in the returned dictionary

            If 'nan' then when there is no data, or only NaN data, for an (x,y) value then this (x,y)
            value will be a key in the returned dictionary and its value will be NaN.

            If 'min' then when there is no data, or only NaN data, for an (x,y) value then this (x,y)
            value will be a key in the returned dictionary and its value will be the minimal value 
            allowed for this statistic, as specified by `lower_cutoff`.
        Returns
        -------
        dict
        """
        vb = {}
        assert(no_data_action in ('discard', 'nan', 'min')), "`no_data_action` must be `discard`, `nan` or `min`!"
        df = self.FilteredDataFrame
        for x in self.x_values:
            for y in self.y_values:
                if statistic == 'monotonic_min':
                    tempdf_xy = df[df[self.x_axis] <= x][df[self.y_axis] <= y]
                elif statistic == 'monotonic_max':
                    tempdf_xy = df[df[self.x_axis] >= x][df[self.y_axis] >= y]
                else:
                    tempdf_xy = df[df[self.x_axis] == x][df[self.y_axis] == y]
                if all(_np.isnan(tempdf_xy[metric])) or len(tempdf_xy[metric]) == 0:
                    if no_data_action == 'discard':
                        pass
                    elif no_data_action == 'min':
                        vb[x, y] = lower_cutoff
                    elif lower_cutoff == 'nan':
                        vb[x, y] = _np.nan
                else:
                    vb[x, y] = _calculate_summary_statistic(tempdf_xy[metric], statistic, lower_cutoff=lower_cutoff)
                
        return vb

    def get_capability_regions(self, metric='polarization', threshold=1 / _np.e, significance=0.05, monotonic=True,
                               no_data_action='discard'):
        """

        no_data_action:
            If 'discard' then when there is no data, or only NaN data, for an (x,y) value then this (x,y)
            value will not be a key in the returned dictionary

            If 'nan' then when there is no data, or only NaN data, for an (x,y) value then this (x,y)
            value will be a key in the returned dictionary and its value will be NaN.
        Returns
        -------
        """
        capreg = {}
        assert(metric in ('polarization', 'success_probability'))

        for x in self.x_values:
            tempdf_x = self.FilteredDataFrame[self.FilteredDataFrame[self.x_axis] == x]
            for y in self.y_values:
                tempdf_xy = tempdf_x[tempdf_x[self.y_axis] == y]
                if metric == 'polarization':
                    #print('rescaling threshold!')
                    assert(len(set(tempdf_xy['Width'])) <= 1)
                    if len(set(tempdf_xy['Width'])) == 1:
                        w = list(tempdf_xy['Width'])[0]
                        sp_threshold = polarization_to_success_probability(threshold, w)
                        #print(w, threshold, sp_threshold)
                    else:
                        # Doesn't matter what the threshold is --- there's no data.
                        sp_threshold = 0
                else:
                    sp_threshold = threshold
                #print(x, y, sp_threshold, threshold)
 
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

        if no_data_action == 'discard':
            trimmed_capreg = {}
            for key, val in capreg.items():
                if not _np.isnan(val):
                    trimmed_capreg[key] = val
            capreg = trimmed_capreg

        return capreg

    # def get_capability_region_boundaries(self, metric='polarization', threshold=1 / _np.e, significance=0.05):

    #     capregion = self.get_capability_regions(metric=metric, threshold=threshold, significance=significance, monotonic=True)
    #     boundaries = {'success':{}, 'fail':{}}
    #     for xval in self.x_values:
    #         print(xval)
    #         boundaries['success'][xval] = _np.max([0] + [y for (x, y), val in capregion.items() if val >= 2 and x == xval])
    #         boundaries['fail'][xval] = _np.max([0] + [y for (x, y), val in capregion.items() if val >= 1 and x == xval])
    #         print(boundaries['success'])
    #         print([0] + [y for (x, y), val in capregion.items() if val >= 2 and x == xval])
    #         print(boundaries['fail'])
    #     return boundaries
