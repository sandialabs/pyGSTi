
from ..util import BaseCase
from pygsti.algorithms.modelselection
from pygsti.tools.modelselectiontools import create_projector_matrix_from_trace, do_greedy_from_full_fast
class AMSTester(BaseCase):
	
	def check_approx_AMS(self, model, data, er_thresh, params_to_keep, options):

		trace, _ = do_greedy_from_full_fast(model, data, er_thresh, *options)
		final_fit_vec = trace[-1][0][0]
		# a trace contains the path taken through model space. We can use this
		#to construct a matrix that converts the full model vector into the 
		#reduced model vector
		reducer = create_projector_matrix_from_trace

		#multiplying the reducer and its transpose creates a
		#diagonal matrix whose entries are either 0 or 1,
		#where entries equal to 0 represent parameters that
		#were removed
		projector_matrix  = reducer.T @ reducer
		for level in trace:
			assert level[0][1]
		for param in params_to_keep:
			assert projector_matrix[param][param] == 1, 'AMS removed a non-trivial parameter'
		
		percent_removed = (model.num_params - len(final_fit_vec)) / (model.num_params - params_to_keep) * 100

		assert percent_removed > 50, 'AMS removed less than 50% of trivial parameters'

	def test_single_errgen_fogi_noise(self):
		from pygsti.modelpacks import smq1Q_XY
		fogi_model
		for i, row in enumerate(fogi_model.param_interposer.inv_transform_matrix):
            non_zero_elems = 0
            last_nonzero = -1
            for j,elem in enumerate(row):
                if np.abs(elem) > 1e-10:
                        non_zero_elems += 1
                        last_nonzero = j
            if non_zero_elems == 1:
                    if 'Hamiltonian' in labels[i] or 'diagonal' in labels[i]:
                        single_errgen_fogis.append([i,last_nonzero])

    errgens_to_test = single_errgen_fogis[:2]