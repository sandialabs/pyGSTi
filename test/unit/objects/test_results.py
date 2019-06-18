import pickle

from ..util import BaseCase, Namespace
from . import fixtures as pkg

from pygsti.objects import results, estimate
from pygsti.construction import std1Q_XYI as std


class ResultsTester(BaseCase):
    def test_results(self):
        # TODO break apart into unit tests
        #prepare a results object
        model = pkg.mdl_lsgst_go
        ds = pkg.dataset
        maxLengthList = pkg.maxLengthList
        gss = pkg.lsgstStructs

        #init results
        res = results.Results()

        with self.assertRaises(ValueError):
            res.add_estimate(None, None, None, None)  # dataset not init yet
        res.init_dataset(ds)

        with self.assertRaises(ValueError):
            res.add_estimate(None, None, None, None)  # gss not init yet
        res.init_circuits(gss)

        with self.assertWarns(Warning):
            res.init_dataset(ds)  # usually don't want to re-init
        with self.assertWarns(Warning):
            res.init_circuits(gss)  # usually don't want to re-init
        with self.assertRaises(ValueError):
            res.init_circuits("foobar")
        res.init_circuits(gss)  # make sure "foobar" test above doesn't leave results in an un-init state

        #add estimates
        res.add_estimate(std.target_model(), std.target_model(),
                         [model] * len(maxLengthList), parameters={'objective': 'logl'},
                         estimate_key="default")
        with self.assertWarns(Warning):
            res.add_estimate(std.target_model(), std.target_model(),
                             [model] * len(maxLengthList), parameters={'objective': 'logl'},
                             estimate_key="default")  # re-init existing estimate

        #Estimate & results render as str
        print(str(res))

        #Results views
        rview = res.view(['default'])
        rview2 = res.view('default')  # this works too

        # add_estimates from other results
        res2 = results.Results()

        with self.assertRaises(ValueError):
            res2.add_estimates(res, ['default'])  # ds not init yet...
        res2.init_dataset(ds)

        with self.assertRaises(ValueError):
            res2.add_estimates(res, ['default'])  # gss not init yet...
        res2.init_circuits(gss)

        res2.add_estimates(res, ['default'])  # now it works!
        with self.assertWarns(Warning):
            res2.add_estimates(res, ['default'])  # b/c re-init exising estimate

        # rename estimates
        res2.rename_estimate('default', 'renamed_default')
        with self.assertRaises(KeyError):
            res2.rename_estimate('foobar', 'renamed_foobar')

        # add estimate from model test
        mdl_guess = std.target_model().depolarize(op_noise=0.07, spam_noise=0.03)
        res2.add_model_test(std.target_model(), mdl_guess, estimate_key='Test', gauge_opt_keys="auto")

        chi2_res = results.Results()
        chi2_res.init_dataset(ds)
        chi2_res.init_circuits(gss)
        chi2_res.add_estimate(std.target_model(), std.target_model(),
                              [model] * len(maxLengthList), parameters={'objective': 'chi2'},
                              estimate_key="default")
        chi2_res.add_model_test(std.target_model(), mdl_guess, estimate_key='Test', gauge_opt_keys="auto")

        chi2_res.estimates['default'].parameters['objective'] = "foobar"  # sets up error below
        chi2_res.estimates['Test'].parameters['objective'] = "foobar"
        #print("DB: ",chi2_res.estimates.keys())
        with self.assertRaises(ValueError):
            chi2_res.add_model_test(std.target_model(), mdl_guess,
                                    estimate_key='Test', gauge_opt_keys="auto")  # invalid "objective"

    def test_results_warns_on_deprecated(self):
        #deprecated functions that issue warnings
        res = results.Results()
        with self.assertWarns(Warning):
            res.create_full_report_pdf()
        with self.assertWarns(Warning):
            res.create_brief_report_pdf()
        with self.assertWarns(Warning):
            res.create_presentation_pdf()
        with self.assertWarns(Warning):
            res.create_presentation_ppt()
        with self.assertWarns(Warning):
            res.create_general_report_pdf()
