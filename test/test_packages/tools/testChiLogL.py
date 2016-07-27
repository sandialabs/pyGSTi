from .toolsBaseCase import ToolsTestCase
from pygsti.construction import std1Q_XYI as std
import pygsti
import unittest


class Chi2LogLTestCase(ToolsTestCase):
    ###########################################################
    ## Chi2 and logL TESTS   ##################################
    ###########################################################

    def test_chi2_fn(self):
        ds = pygsti.objects.DataSet(fileToLoadFrom="../cmp_chk_files/analysis.dataset")
        chi2, grad = pygsti.chi2(ds, std.gs_target, returnGradient=True)

        pygsti.gate_string_chi2( ('Gx',), ds, std.gs_target)
        pygsti.chi2fn_2outcome( N=100, p=0.5, f=0.6)
        pygsti.chi2fn_2outcome_wfreqs( N=100, p=0.5, f=0.6)
        pygsti.chi2fn( N=100, p=0.5, f=0.6)
        pygsti.chi2fn_wfreqs( N=100, p=0.5, f=0.6)

        with self.assertRaises(ValueError):
            pygsti.chi2(ds, std.gs_target, useFreqWeightedChiSq=True) #no impl yet

    def test_logl_fn(self):
        ds = pygsti.objects.DataSet(fileToLoadFrom="../cmp_chk_files/analysis.dataset")
        gatestrings = pygsti.construction.gatestring_list( [ ('Gx',), ('Gy',), ('Gx','Gx') ] )
        spam_labels = std.gs_target.get_spam_labels()
        pygsti.create_count_vec_dict( spam_labels, ds, gatestrings )

        L1 = pygsti.logl(std.gs_target, ds, gatestrings,
                         probClipInterval=(-1e6,1e6), countVecMx=None,
                         poissonPicture=True, check=False)
        L2 = pygsti.logl(std.gs_target, ds, gatestrings,
                         probClipInterval=(-1e6,1e6), countVecMx=None,
                         poissonPicture=False, check=False) #Non-poisson-picture

        dL1 = pygsti.logl_jacobian(std.gs_target, ds, gatestrings,
                                   probClipInterval=(-1e6,1e6), radius=1e-4,
                                   poissonPicture=True, check=False)
        dL2 = pygsti.logl_jacobian(std.gs_target, ds, gatestrings,
                                   probClipInterval=(-1e6,1e6), radius=1e-4,
                                   poissonPicture=False, check=False)
        dL2b = pygsti.logl_jacobian(std.gs_target, ds, None,
                                   probClipInterval=(-1e6,1e6), radius=1e-4,
                                   poissonPicture=False, check=False) #test None as gs list


        hL1 = pygsti.logl_hessian(std.gs_target, ds, gatestrings,
                                  probClipInterval=(-1e6,1e6), radius=1e-4,
                                  poissonPicture=True, check=False)

        hL2 = pygsti.logl_hessian(std.gs_target, ds, gatestrings,
                                  probClipInterval=(-1e6,1e6), radius=1e-4,
                                  poissonPicture=False, check=False)
        hL2b = pygsti.logl_hessian(std.gs_target, ds, None,
                                   probClipInterval=(-1e6,1e6), radius=1e-4,
                                   poissonPicture=False, check=False) #test None as gs list


        maxL1 = pygsti.logl_max(ds, gatestrings, poissonPicture=True, check=True)
        maxL2 = pygsti.logl_max(ds, gatestrings, poissonPicture=False, check=True)

        pygsti.cptp_penalty(std.gs_target, include_spam_penalty=True)
        twoDelta1 = pygsti.two_delta_loglfn(N=100, p=0.5, f=0.6, minProbClip=1e-6, poissonPicture=True)
        twoDelta2 = pygsti.two_delta_loglfn(N=100, p=0.5, f=0.6, minProbClip=1e-6, poissonPicture=False)

if __name__ == '__main__':
    unittest.main(verbosity=2)
