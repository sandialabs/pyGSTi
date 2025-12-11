
from ..util import BaseCase
from pygsti.algorithms.modelselection import do_greedy_from_full_fast
from pygsti.tools.modelselectiontools import create_projector_matrix_from_trace
import numpy as _np
from pygsti.data import simulate_data
from pygsti.protocols import ProtocolData
class AMSTester(BaseCase):
	
	def check_approx_AMS(self, model, data, er_thresh, params_to_keep, options):

		trace = do_greedy_from_full_fast(model, data, er_thresh, *options)
		final_fit_vec = trace[-1][0]
		# a trace contains the path taken through model space. We can use this
		#to construct a matrix that converts the full model vector into the 
		#reduced model vector
		reducer = create_projector_matrix_from_trace(trace).T

		#multiplying the reducer and its transpose creates a
		#diagonal matrix whose entries are either 0 or 1,
		#where entries equal to 0 represent parameters that
		#were removed
		projector_matrix  = reducer.T @ reducer

		for param in params_to_keep:
			assert projector_matrix[param][param] == 1, 'AMS removed a non-trivial parameter'
		
		percent_removed = (model.num_params - len(final_fit_vec)) / (model.num_params - len(params_to_keep)) * 100

		assert percent_removed > 50, f'AMS removed less than 50% of trivial parameters, {model.num_params=}, {len(final_fit_vec)=}'

	def test_single_errgen_fogi_noise(self):
		from pygsti.modelpacks import smq1Q_XY, smq1Q_XYZI
		packs_to_test = [smq1Q_XY, smq1Q_XYZI]
		for pack in packs_to_test:
			datagen_model = pack.target_model('GLND')
			edesign = pack.create_gst_experiment_design(max_max_length=4)
			datagen_model.depolarize(spam_noise=1e-5)

			fogi_model = pack.target_model('FOGI-GLND')
			labels = fogi_model.parameter_labels

			#First we identify FOGI quantities that have a single error generator in them
			single_errgen_fogis = []
			for i, row in enumerate(fogi_model.param_interposer.inv_transform_matrix):
				non_zero_elems = 0
				last_nonzero = -1
				for j,elem in enumerate(row):
					if _np.abs(elem) > 1e-10:
							non_zero_elems += 1
							last_nonzero = j
				if non_zero_elems == 1:
						#To ensure the generated models are physical, we only focus on
						#hamiltonian and stochastic errors.
						if 'Hamiltonian' in labels[i] or 'diagonal' in labels[i]:
							single_errgen_fogis.append([i,last_nonzero])
			#We don't want the test to take too long, so we only test
			#two of them
			errgens_to_test = single_errgen_fogis[:2]

			for k, (fogi_index, errgen_index) in enumerate(errgens_to_test):

				error_vec_copy = datagen_model.to_vector().copy()
				error_value = .005
				
				error_vec_copy[errgen_index] = error_value
				datagen_model.from_vector(error_vec_copy)
				dataset = simulate_data(datagen_model, edesign.all_circuits_needing_data, 
												num_samples=10000, seed=20230217)
				data = ProtocolData(edesign, dataset)
				self.check_approx_AMS(fogi_model, data, 2, [fogi_index], [2, 300, 1e-12, 1e-4, 200, True, None, None,None])