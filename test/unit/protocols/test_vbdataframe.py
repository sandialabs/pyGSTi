from ..util import BaseCase

import pygsti
import pandas
import os
import numpy as np
from pygsti.protocols import vbdataframe as _vbdataframe

class TestVBDataFrame(BaseCase):

    def test_vbdataframe_1(self):
        
        df = pandas.read_csv(os.path.join(os.path.dirname(__file__),'test_dataframe_1.csv'))

        depths = [0, 4, 8, 12, 20, 28, 40, 56, 80, 112, 160, 224, 316]
        vbdf = _vbdataframe.VBDataFrame(df, x_values=depths)

        # ///// Tests filtering data ///// #
        vbdf1 = vbdf.filter_data('Qubits', metric='polarization', statistic='mean', threshold=1/np.e,
                                 indep_x=False)
        # Nothing in this test checks that the correct output is produced here.
        vbdf2 = vbdf.filter_data('Qubits', metric='success_probabilities', statistic='max', threshold=1/np.e,
                                 indep_x=True)  

        qubits = set(vbdf1.dataframe['Qubits'])
        # Checks the correct qubits are selected.
        qubits == {"('Q0', 'Q1')", "('Q0', 'Q1', 'Q2', 'Q3', 'Q4')", "('Q0', 'Q1', 'Q2', 'Q4')", "('Q0',)"}

        # ///// Tests extract VB data ///// #
        vbdata1 = vbdf1.vb_data(metric='polarization', statistic='mean', lower_cutoff=0., no_data_action='discard')

        correct_vbdata1 = {(0, 1): 0.851416015625,
                         (0, 2): 0.7064453125,
                         (0, 4): 0.5225,
                         (0, 5): 0.351335685483871,
                         (4, 1): 0.822509765625,
                         (4, 2): 0.6943359375,
                         (4, 4): 0.44955729166666664,
                         (4, 5): 0.3045866935483871,
                         (8, 1): 0.830419921875,
                         (8, 2): 0.6907877604166666,
                         (8, 4): 0.42536458333333327,
                         (8, 5): 0.2916834677419355,
                         (12, 1): 0.838671875,
                         (12, 2): 0.6174479166666667,
                         (12, 4): 0.4035677083333333,
                         (12, 5): 0.25471270161290327,
                         (20, 1): 0.796875,
                         (20, 2): 0.6025065104166667,
                         (20, 4): 0.33934895833333334,
                         (20, 5): 0.21519657258064515,
                         (28, 1): 0.765869140625,
                         (28, 2): 0.5246744791666667,
                         (28, 4): 0.2816927083333334,
                         (28, 5): 0.16771673387096775,
                         (40, 1): 0.775,
                         (40, 2): 0.45244140625,
                         (40, 4): 0.18473958333333332,
                         (40, 5): 0.12220262096774193,
                         (80, 1): 0.659375,
                         (80, 2): 0.3126953125,
                         (80, 4): 0.058776041666666654,
                         (80, 5): 0.030317540322580643}

        self.assertTrue(set(correct_vbdata1.keys()) == set(vbdata1.keys()))

        for key in vbdata1.keys():
            self.assertAlmostEqual(vbdata1[key], correct_vbdata1[key])

        # Nothing in this test checks that the correct output is produced here.
        vbdata2 = vbdf1.vb_data(metric='success_probabilities', statistic='max', lower_cutoff=0., no_data_action='nan')
        vbdata3 = vbdf1.vb_data(metric='success_probabilities', statistic='min', lower_cutoff=0.1, no_data_action='min')
        vbdata4 = vbdf1.vb_data(metric='success_probabilities', statistic='monotonic_min', lower_cutoff=0., no_data_action='discard')
        vbdata5 = vbdf1.vb_data(metric='success_probabilities', statistic='monotonic_max', lower_cutoff=0., no_data_action='discard')


    def test_vbdataframe_2(self):
        
        df = pandas.read_csv(os.path.join(os.path.dirname(__file__),'test_dataframe_2.csv'))

        vbdf = _vbdataframe.VBDataFrame(df)

        vbdf1 = vbdf.select_column_value('Pass', 'pass1')
        
        capreg1 = vbdf1.capability_regions(metric='polarization', threshold=1/np.e, significance=0.05, monotonic=True,
                                        nan_data_action='discard')

        correct_capreg1 = {(0, 1): 2,
                             (0, 2): 2,
                             (0, 3): 2,
                             (0, 4): 2,
                             (0, 5): 2,
                             (4, 1): 2,
                             (4, 2): 2,
                             (4, 3): 2,
                             (4, 4): 2,
                             (4, 5): 1,
                             (8, 1): 2,
                             (8, 2): 2,
                             (8, 3): 2,
                             (8, 4): 2,
                             (8, 5): 1,
                             (16, 1): 2,
                             (16, 2): 2,
                             (16, 3): 2,
                             (16, 4): 1,
                             (16, 5): 1,
                             (32, 1): 2,
                             (32, 2): 2,
                             (32, 3): 1,
                             (32, 4): 1,
                             (32, 5): 1,
                             (64, 1): 2,
                             (64, 2): 1,
                             (64, 3): 1,
                             (64, 4): 1,
                             (64, 5): 0,
                             (128, 1): 2,
                             (128, 2): 1,
                             (128, 3): 1,
                             (128, 4): 0,
                             (128, 5): 0,
                             (256, 1): 2,
                             (256, 2): 1,
                             (256, 3): 0,
                             (256, 4): 0,
                             (256, 5): 0,
                             (512, 1): 1,
                             (512, 2): 0,
                             (512, 3): 0}

        self.assertTrue(set(correct_capreg1.keys()) == set(capreg1.keys()))

        for key in capreg1.keys():
            self.assertAlmostEqual(capreg1[key], correct_capreg1[key])

        capreg2 = vbdf1.capability_regions(metric='success_probability', threshold=2/3, significance=0.01, monotonic=False,
                                 nan_data_action='none')
