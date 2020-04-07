
from ..testutils import BaseTestCase, temp_files, compare_files
import unittest
import pygsti as gst
import numpy as np
from pygsti.modelpacks.legacy import std1Q_XY as Std1Q_XY

from pygsti.extras import rpe
RPE = rpe
RPEConstr = rpe.rpeconstruction
rpeconfig_GxPi2_GyPi2_00 = rpe.rpeconfig_GxPi2_GyPi2_00



class TestRPEMethods(BaseTestCase):

    def test_rpe_demo(self):

        #Declare the particular RPE instance we are interested in
        #(X and Y pi/2 rotations)
        rpeconfig_inst = rpeconfig_GxPi2_GyPi2_00
        
        
        #Declare a variety of relevant parameters
        target_model = Std1Q_XY.target_model()
        target_model.set_all_parameterizations('TP')
        maxLengths_1024 = [1,2,4,8,16,32,64,128,256,512,1024]
        
        stringListsRPE = RPEConstr.make_rpe_angle_string_list_dict(10,rpeconfig_inst)
        
        angleList = ['alpha','epsilon','theta']
        
        numStrsD = {}
        numStrsD['RPE'] = [6*i for i in np.arange(1,12)]
        
        #Create noisy model
        mdl_real = target_model.rotate(rotate=[0.01,0.01,0])
        
        #Extract noisy model angles
        true_alpha = RPE.extract_alpha(mdl_real,rpeconfig_inst)
        true_epsilon = RPE.extract_epsilon(mdl_real,rpeconfig_inst)
        true_theta = RPE.extract_theta(mdl_real,rpeconfig_inst)
        
        #Load pre-simulated dataset
#        N=100
#        DS = gst.construction.generate_fake_data(mdl_real,stringListsRPE['totalStrList'],N,sampleError='binomial',seed=1)
        DS = gst.io.load_dataset(compare_files + '/rpe_test_ds.txt')
        
        #Analyze dataset
        resultsRPE = RPE.analyze_rpe_data(DS,mdl_real,stringListsRPE,rpeconfig_inst)
    
        PhiFunErrorListCorrect = np.array([1.4647120176458639e-08, 5.466086847039087e-09, 2.811838817340373e-09, 9.295340015064157e-09, 1.4896280285670027e-08, 1.4897848815698111e-08, 4.269122493016919e-09, 1.4897576120637135e-08, 1.4897610849801124e-08, 6.193216574995608e-09, 1.4469989279702888e-08])
        alphaErrorListCorrect= np.array([0.05000352128724339, 0.09825031832409103, 0.02500687294425541, 0.012575500499770742, 0.012523502109159201, 0.0044641536173215535, 0.0007474956215971496, 0.00018069665046693828, 0.00027322234186732963, 0.00020451259672338296, 3.198565800954789e-05])
        epsilonErrorListCorrect  = np.array([0.18811777239515082, 0.009964509397691668, 0.004957204616348632, 0.007362158521305728, 0.00010888027730326932, 0.0015920480408759818, 0.001403238695757869, 0.0004870373015233298, 0.0001929699810709895, 3.411170328226909e-05, 2.723356656519904e-05]) 
        thetaErrorListCorrect= np.array([0.018281791956737087, 0.015230174647994477, 0.0018336710008779447, 0.004525418577473875, 0.0047631900047339125, 0.002627347622582976, 0.0030228260649800788, 0.002591470061459089, 0.0027097752869584, 0.002733081374122569, 0.0027947590038843876])

        PhiFunErrorList =  resultsRPE['PhiFunErrorList']
        alphaErrorList =  resultsRPE['alphaErrorList']
        epsilonErrorList =  resultsRPE['epsilonErrorList']
        thetaErrorList =  resultsRPE['thetaErrorList']
        
        assert np.linalg.norm(PhiFunErrorListCorrect-PhiFunErrorList) < 1e-8
        assert np.linalg.norm(alphaErrorListCorrect-alphaErrorList) < 1e-8
        assert np.linalg.norm(epsilonErrorListCorrect-epsilonErrorList) < 1e-8
        assert np.linalg.norm(thetaErrorListCorrect-thetaErrorList) < 1e-8
        
        
        # again, with consistency checks
#       We are currently not testing the consistency check. -KMR 2/26/18

#        dummy_k_list = [ 1 ] #EGN: not sure what this should really be...
#        resultsRPE_2 = RPE.analyze_rpe_data(DS,mdl_real,stringListsRPE,rpeconfig_inst,
#                                            do_consistency_check=True, k_list=dummy_k_list)



#        with self.assertRaises(ValueError):
#            RPE.analyze_rpe_data(DS,mdl_real,stringListsRPE,rpeconfig_inst,
#                                 do_consistency_check=True) #no k_list given
        



        #Print results
        print('alpha_true - pi/2 =',true_alpha-np.pi/2)
        print('epsilon_true - pi/2 =',true_epsilon-np.pi/2)
        print('theta_true =',true_theta)
        print()
        print('alpha_true - alpha_est_final =',resultsRPE['alphaErrorList'][-1])
        print('epsilon_true - epsilon_est_final =',resultsRPE['epsilonErrorList'][-1])
        print('theta_true - theta_est_final =',resultsRPE['thetaErrorList'][-1])

