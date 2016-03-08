#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation              
#    This Software is released under the GPL license detailed    
#    in the file "license.txt" in the top-level pyGSTi directory 
#*****************************************************************
""" Defines the rpeconfig class and supporting functionality."""

import numpy as _np

class rpeconfig:
    """
    Encapsulates a collection of settings for an RPE run.  Provides full specifications,
    including target gates, SPAM, sine and cosine strings for alpha, epsilon, and theta. 
    """
    
    def __init__(self,input_dict):
        unspecified_keys = []
        try:
            self.fixed_axis_gate_label = input_dict['fixed_axis_gate_label']
        except:
            unspecified_keys.append('fixed_axis_gate_label')
        try:
            self.fixed_axis_label = input_dict['fixed_axis_label']
        except:
            unspecified_keys.append('fixed_axis_label')
        try:
            self.fixed_axis_target = input_dict['fixed_axis_target']
        except:
            unspecified_keys.append('fixed_axis_target')
        try:
            self.loose_axis_gate_label = input_dict['loose_axis_gate_label']
        except:
            unspecified_keys.append('loose_axis_gate_label')
        try:
            self.loose_axis_label = input_dict['loose_axis_label']
        except:
            unspecified_keys.append('loose_axis_label')
        try:
            self.loose_axis_target = input_dict['loose_axis_target']
        except:
            unspecified_keys.append('loose_axis_target')
        try:
            self.auxiliary_axis_gate_label = input_dict['auxiliary_axis_gate_label']
        except:
            unspecified_keys.append('auxiliary_axis_gate_label')
        try:
            self.auxiliary_axis_label = input_dict['auxiliary_axis_label']
        except:
            unspecified_keys.append('auxiliary_axis_label')
        try:
            self.rhoExpressions = input_dict['rhoExpressions']
        except:
            unspecified_keys.append('rhoExpressions')
        try:
            self.EExpressions = input_dict['EExpressions']
        except:
            unspecified_keys.append('EExpressions')
        try:
            self.spamLabelDict = input_dict['spamLabelDict']
        except:
            unspecified_keys.append('spamLabelDict')
        try:
            self.alpha = input_dict['alpha']
        except:
            unspecified_keys.append('alpha')
        try:
            self.epsilon = input_dict['epsilon']
        except:
            unspecified_keys.append('epsilon')
        try:
            self.theta = input_dict['theta']
        except:
            unspecified_keys.append('theta')
        try:
            self.new_epsilon_func = input_dict['new_epsilon_func']
        except:
            unspecified_keys.append('new_epsilon_func')
        try:
            self.alpha_hat_func = input_dict['alpha_hat_func']
        except:
            unspecified_keys.append('alpha_hat_func')
        try:
            self.epsilon_hat_func = input_dict['epsilon_hat_func']
        except:
            unspecified_keys.append('epsilon_hat_func')
        try:
            self.Phi_hat_func = input_dict['Phi_hat_func']
        except:
            unspecified_keys.append('Phi_hat_func')
        try:
            self.alpha_cos_prep_tuple = input_dict['alpha_cos_prep_tuple']
        except:
            unspecified_keys.append('alpha_cos_prep_tuple')
        try:
            self.alpha_cos_prep_str = input_dict['alpha_cos_prep_str']
        except:
            unspecified_keys.append('alpha_cos_prep_str')
        try:
            self.alpha_cos_germ_tuple = input_dict['alpha_cos_germ_tuple']
        except:
            unspecified_keys.append('alpha_cos_germ_tuple')
        try:
            self.alpha_cos_germ_str = input_dict['alpha_cos_germ_str']
        except:
            unspecified_keys.append('alpha_cos_germ_str')
        try:
            self.alpha_cos_meas_tuple = input_dict['alpha_cos_meas_tuple']
        except:
            unspecified_keys.append('alpha_cos_meas_tuple')
        try:
            self.alpha_cos_meas_str = input_dict['alpha_cos_meas_str']
        except:
            unspecified_keys.append('alpha_cos_meas_str')
        try:
            self.alpha_sin_prep_tuple = input_dict['alpha_sin_prep_tuple']
        except:
            unspecified_keys.append('alpha_sin_prep_tuple')
        try:
            self.alpha_sin_prep_str = input_dict['alpha_sin_prep_str']
        except:
            unspecified_keys.append('alpha_sin_prep_str')
        try:
            self.alpha_sin_germ_tuple = input_dict['alpha_sin_germ_tuple']
        except:
            unspecified_keys.append('alpha_sin_germ_tuple')
        try:
            self.alpha_sin_germ_str = input_dict['alpha_sin_germ_str']
        except:
            unspecified_keys.append('alpha_sin_germ_str')
        try:
            self.alpha_sin_meas_tuple = input_dict['alpha_sin_meas_tuple']
        except:
            unspecified_keys.append('alpha_sin_meas_tuple')
        try:
            self.alpha_sin_meas_str = input_dict['alpha_sin_meas_str']
        except:
            unspecified_keys.append('alpha_sin_meas_str')
        try:
            self.epsilon_cos_prep_tuple = input_dict['epsilon_cos_prep_tuple']
        except:
            unspecified_keys.append('epsilon_cos_prep_tuple')
        try:
            self.epsilon_cos_prep_str = input_dict['epsilon_cos_prep_str']
        except:
            unspecified_keys.append('epsilon_cos_prep_str')
        try:
            self.epsilon_cos_germ_tuple = input_dict['epsilon_cos_germ_tuple']
        except:
            unspecified_keys.append('epsilon_cos_germ_tuple')
        try:
            self.epsilon_cos_germ_str = input_dict['epsilon_cos_germ_str']
        except:
            unspecified_keys.append('epsilon_cos_germ_str')
        try:
            self.epsilon_cos_meas_tuple = input_dict['epsilon_cos_meas_tuple']
        except:
            unspecified_keys.append('epsilon_cos_meas_tuple')
        try:
            self.epsilon_cos_meas_str = input_dict['epsilon_cos_meas_str']
        except:
            unspecified_keys.append('epsilon_cos_meas_str')
        try:
            self.epsilon_sin_prep_tuple = input_dict['epsilon_sin_prep_tuple']
        except:
            unspecified_keys.append('epsilon_sin_prep_tuple')
        try:
            self.epsilon_sin_prep_str = input_dict['epsilon_sin_prep_str']
        except:
            unspecified_keys.append('epsilon_sin_prep_str')
        try:
            self.epsilon_sin_germ_tuple = input_dict['epsilon_sin_germ_tuple']
        except:
            unspecified_keys.append('epsilon_sin_germ_tuple')
        try:
            self.epsilon_sin_germ_str = input_dict['epsilon_sin_germ_str']
        except:
            unspecified_keys.append('epsilon_sin_germ_str')
        try:
            self.epsilon_sin_meas_tuple = input_dict['epsilon_sin_meas_tuple']
        except:
            unspecified_keys.append('epsilon_sin_meas_tuple')
        try:
            self.epsilon_sin_meas_str = input_dict['epsilon_sin_meas_str']
        except:
            unspecified_keys.append('epsilon_sin_meas_str')
        try:
            self.theta_cos_prep_tuple = input_dict['theta_cos_prep_tuple']
        except:
            unspecified_keys.append('theta_cos_prep_tuple')
        try:
            self.theta_cos_prep_str = input_dict['theta_cos_prep_str']
        except:
            unspecified_keys.append('theta_cos_prep_str')
        try:
            self.theta_cos_germ_tuple = input_dict['theta_cos_germ_tuple']
        except:
            unspecified_keys.append('theta_cos_germ_tuple')
        try:
            self.theta_cos_germ_str = input_dict['theta_cos_germ_str']
        except:
            unspecified_keys.append('theta_cos_germ_str')
        try:
            self.theta_cos_meas_tuple = input_dict['theta_cos_meas_tuple']
        except:
            unspecified_keys.append('theta_cos_meas_tuple')
        try:
            self.theta_cos_meas_str = input_dict['theta_cos_meas_str']
        except:
            unspecified_keys.append('theta_cos_meas_str')
        try:
            self.theta_sin_prep_tuple = input_dict['theta_sin_prep_tuple']
        except:
            unspecified_keys.append('theta_sin_prep_tuple')
        try:
            self.theta_sin_prep_str = input_dict['theta_sin_prep_str']
        except:
            unspecified_keys.append('theta_sin_prep_str')
        try:
            self.theta_sin_germ_tuple = input_dict['theta_sin_germ_tuple']
        except:
            unspecified_keys.append('theta_sin_germ_tuple')
        try:
            self.theta_sin_germ_str = input_dict['theta_sin_germ_str']
        except:
            unspecified_keys.append('theta_sin_germ_str')
        try:
            self.theta_sin_meas_tuple = input_dict['theta_sin_meas_tuple']
        except:
            unspecified_keys.append('theta_sin_meas_tuple')
        try:
            self.theta_sin_meas_str = input_dict['theta_sin_meas_str']
        except:
            unspecified_keys.append('theta_sin_meas_str')
        if unspecified_keys == []:
            print "Fully specified RPE configuration."
        else:
            print "RPE configuration not fully specified.  Missing following keys:"
            for key in unspecified_keys:
                print key
        
