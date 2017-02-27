from __future__ import print_function

from ..testutils import BaseTestCase
import unittest
import pygsti as gst
import numpy as np
import pygsti.construction.std1Q_XY as Std1Q_XY

from pygsti.extras import rpe
RPE = rpe
RPEConstr = rpe.rpeconstruction
rpeconfig_GxPi2_GyPi2_UpDn = rpe.rpeconfig_GxPi2_GyPi2_UpDn

class TestRPEMethods(BaseTestCase):

    def test_rpe_demo(self):

        #Declare the particular RPE instance we are interested in
        #(X and Y pi/2 rotations)
        rpeconfig_inst = rpeconfig_GxPi2_GyPi2_UpDn
        
        
        #Declare a variety of relevant parameters
        gs_target = Std1Q_XY.gs_target
        gs_target.set_all_parameterizations('TP')
        maxLengths_1024 = [1,2,4,8,16,32,64,128,256,512,1024]
        
        stringListsRPE = RPEConstr.make_rpe_angle_string_list_dict(10,rpeconfig_inst)
        
        angleList = ['alpha','epsilon','theta']
        
        numStrsD = {}
        numStrsD['RPE'] = [6*i for i in np.arange(1,12)]
        
        #Create noisy gateset
        gs_real = gs_target.randomize_with_unitary(.01,seed=0)
        
        #Extract noisy gateset angles
        true_alpha = RPE.extract_alpha(gs_real,rpeconfig_inst)
        true_epsilon = RPE.extract_epsilon(gs_real,rpeconfig_inst)
        true_theta = RPE.extract_theta(gs_real,rpeconfig_inst)
        
        #Simulate dataset
        N=100
        DS = gst.construction.generate_fake_data(gs_real,stringListsRPE['totalStrList'],N,sampleError='binomial',seed=1)
        
        #Analyze dataset
        resultsRPE = RPE.analyze_rpe_data(DS,gs_real,stringListsRPE,rpeconfig_inst)
        
        #Print results
        print('alpha_true - pi/2 =',true_alpha-np.pi/2)
        print('epsilon_true - pi/2 =',true_epsilon-np.pi/2)
        print('theta_true =',true_theta)
        print()
        print('alpha_true - alpha_est_final =',resultsRPE['alphaErrorList'][-1])
        print('epsilon_true - epsilon_est_final =',resultsRPE['epsilonErrorList'][-1])
        print('theta_true - theta_est_final =',resultsRPE['thetaErrorList'][-1])

