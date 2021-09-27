import unittest
from ..testutils import BaseTestCase, temp_files

import shutil, os

import pygsti
import pygsti.io as io
from pygsti.modelpacks import smq1Q_XYI

from pygsti.processors import QubitProcessorSpec
from pygsti.processors.compilationrules import CliffordCompilationRules as CCR


def clear_root(pth):
    shutil.rmtree(pth)
    os.mkdir(pth)


class NiceSerializationTester(BaseTestCase):

    def test_nice_serialization(self):
        test_serialization_root = temp_files + "/test_serialization_root"
        pspec = pygsti.processors.QubitProcessorSpec(4, ('Gxpi2', 'Gypi2', 'Gcnot'), geometry='line')

        # 1Q GST
        gst_design0 = smq1Q_XYI.get_gst_experiment_design(4, qubit_labels=[0])
        gst_design1 = smq1Q_XYI.get_gst_experiment_design(4, qubit_labels=[1])
        gst_design0.add_default_protocol(pygsti.protocols.StandardGST("full TP"))

        # 2Q Direct RB
        pspec23 = pygsti.processors.QubitProcessorSpec(2, ('Gxpi2', 'Gypi2', 'Gcnot'), geometry='line', qubit_labels=[2,3])
        compilations23 = {'absolute': CCR.create_standard(pspec23, 'absolute', ('paulis', '1Qcliffords'), verbosity=0),
                          'paulieq': CCR.create_standard(pspec23, 'paulieq', ('1Qcliffords', 'allcnots'), verbosity=0)}

        depths = [0, 1, 2, 4, 8]
        k = 10

        rb_design23 = pygsti.protocols.DirectRBDesign(pspec23, compilations23, depths, k)
        rb_design23.add_default_protocol(pygsti.protocols.ByDepthSummaryStatistics(name="VBpolarization"))

        # 4Q Direct RB
        compilations = {'absolute': CCR.create_standard(pspec, 'absolute', ('paulis', '1Qcliffords'), verbosity=0),            
                        'paulieq': CCR.create_standard(pspec, 'paulieq', ('1Qcliffords', 'allcnots'), verbosity=0)}
        rb_design = pygsti.protocols.DirectRBDesign(pspec, compilations, depths, k)

        # Separate and simultaneous designs
        sdesigns_separate = [pygsti.protocols.SimultaneousExperimentDesign([sub_design], qubit_labels=[0,1,2,3])
                             for sub_design in [gst_design0, gst_design1, rb_design23]]

        sdesign_together = pygsti.protocols.SimultaneousExperimentDesign([gst_design0, gst_design1, rb_design23], qubit_labels=[0,1,2,3])

        # Put everything into one combined design
        combined_design = pygsti.protocols.CombinedExperimentDesign({'gst0': sdesigns_separate[0],
                                                                     'gst1': sdesigns_separate[1],
                                                                     'rb23': sdesigns_separate[2],
                                                                     'together': sdesign_together,
                                                                     'rb': rb_design})
        # Test wrtiting just the edesign
        combined_design.write(test_serialization_root)

        # Next, try writing an empty protocol data
        clear_root(test_serialization_root)
        pygsti.io.write_empty_protocol_data(combined_design, test_serialization_root, clobber_ok=True)

        # Generate some fake data
        datagen_model = pygsti.models.create_crosstalk_free_model(pspec, depolarization_strengths={'Gxpi2': 0.03, 'Gypi2': 0.01, 'Gcnot': 0.09})
        pygsti.io.fill_in_empty_dataset_with_fake_data(datagen_model, os.path.join(test_serialization_root, "data/dataset.txt"),
                                                       1000, seed=1234)

        # Load data & edesigns in 
        data = pygsti.io.load_data_from_dir(test_serialization_root)

        # Run protocols & write results
        gst_proto = pygsti.protocols.StandardGST("full TP")
        gst_results = gst_proto.run(data['gst0'][(0,)])

        gst_results.write()

        vb_proto = pygsti.protocols.ByDepthSummaryStatistics(statistics_to_compute=('polarization',))
        vb_results = vb_proto.run(data['rb23'][(2,3)])

        vb_results.write()

        rb_proto = pygsti.protocols.RandomizedBenchmarking()
        rb_results = rb_proto.run(data['rb'])

        rb_results.write()

        # Read in results and look at them
        results = pygsti.io.load_results_from_dir(test_serialization_root)

        rdir = results['gst0'][(0,)]
        rdir.for_protocol['StandardGST'].estimates['full TP'].models

        rdir = results['rb23'][(2,3)]
        r = rdir.for_protocol['ByDepthSummaryStatistics']
        r.to_dataframe()

        r = results['rb'].for_protocol['RandomizedBenchmarking']
        r.fits['full'].estimates
