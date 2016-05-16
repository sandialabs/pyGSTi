import unittest
import pygsti
import numpy as np
import warnings
import pickle


class GateSetOldNewTestCase(unittest.TestCase):

    def setUp(self):
        #OK for these tests, since we test user interface?
        #Set GateSet objects to "strict" mode for testing
        pygsti.objects.GateSet._strict = False

        self.gateset = pygsti.construction.build_gateset(
            [2], [('Q0',)],['Gi','Gx','Gy'], 
            [ "I(Q0)","X(pi/8,Q0)", "Y(pi/8,Q0)"],
            prepLabels=["rho0"], prepExpressions=["0"],
            effectLabels=["E0"], effectExpressions=["1"], 
            spamdefs={'plus': ('rho0','E0'), 
                           'minus': ('rho0','remainder') } )

        self.tp_gateset = pygsti.construction.build_gateset(
            [2], [('Q0',)],['Gi','Gx','Gy'], 
            [ "I(Q0)","X(pi/8,Q0)", "Y(pi/8,Q0)"],
            prepLabels=["rho0"], prepExpressions=["0"],
            effectLabels=["E0"], effectExpressions=["1"], 
            spamdefs={'plus': ('rho0','E0'), 
                           'minus': ('rho0','remainder') },
            parameterization="TP")

        self.static_gateset = pygsti.construction.build_gateset(
            [2], [('Q0',)],['Gi','Gx','Gy'], 
            [ "I(Q0)","X(pi/8,Q0)", "Y(pi/8,Q0)"],
            prepLabels=["rho0"], prepExpressions=["0"],
            effectLabels=["E0"], effectExpressions=["1"], 
            spamdefs={'plus': ('rho0','E0'), 
                           'minus': ('rho0','remainder') },
            parameterization="static")



    def assertArraysAlmostEqual(self,a,b):
        self.assertAlmostEqual( np.linalg.norm(a-b), 0 )

    def assertNoWarnings(self, callable, *args, **kwds):
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter('always')
            result = callable(*args, **kwds)
            self.assertTrue(len(warning_list) == 0)
        return result

    def assertWarns(self, callable, *args, **kwds):
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter('always')
            result = callable(*args, **kwds)
            self.assertTrue(len(warning_list) > 0)
        return result


class TestGateSetOldNewMethods(GateSetOldNewTestCase):

  def test_bulk_multiplication(self):
      gatestring1 = ('Gx','Gy')
      gatestring2 = ('Gx','Gy','Gy')
      evt = self.gateset.bulk_evaltree( [gatestring1,gatestring2] )

      bulk_prods = self.gateset.bulk_product(evt)

      bulk_prods_new = self.gateset._calc().bulk_product_new(evt)
      bulk_prods_scaled_new, scaleVals_new = self.gateset._calc().bulk_product_new(evt, bScale=True)
      bulk_prods2_new = scaleVals_new[:,None,None] * bulk_prods_scaled_new
      self.assertArraysAlmostEqual(bulk_prods,bulk_prods_new)
      self.assertArraysAlmostEqual(bulk_prods,bulk_prods2_new)

      #Artificially reset the "smallness" threshold for scaling to be
      # sure to engate the scaling machinery
      PORIG = pygsti.objects.gscalc.PSMALL; pygsti.objects.gscalc.PSMALL = 10
      bulk_prods_scaled, scaleVals3 = self.gateset.bulk_product(evt, bScale=True)
      bulk_prods3 = scaleVals3[:,None,None] * bulk_prods_scaled
      pygsti.objects.gscalc.PSMALL = PORIG

      bulk_prods_scaled_new, scaleVals3_new = self.gateset._calc().bulk_product_new(evt, bScale=True)
      bulk_prods3_new = scaleVals3_new[:,None,None] * bulk_prods_scaled_new
      self.assertArraysAlmostEqual(bulk_prods3,bulk_prods3_new)
      

  def test_bulk_probabilities(self):
      gatestring1 = ('Gx','Gy')
      gatestring2 = ('Gx','Gy','Gy')
      evt = self.gateset.bulk_evaltree( [gatestring1,gatestring2] )      

      #check == true could raise a warning if a mismatch is detected
      bulk_pr = self.assertNoWarnings(self.gateset.bulk_pr,'plus',evt,check=True)
      bulk_pr_m = self.assertNoWarnings(self.gateset.bulk_pr,'minus',evt,check=True)
      bulk_pr_new = self.assertNoWarnings(self.gateset._calc().bulk_pr_new,'plus',evt,check=True)
      bulk_pr_m_new = self.assertNoWarnings(self.gateset._calc().bulk_pr_new,'minus',evt,check=True)

      self.assertArraysAlmostEqual(bulk_pr,bulk_pr_new)
      self.assertArraysAlmostEqual(bulk_pr_m,bulk_pr_m_new)

      bulk_probs = self.assertNoWarnings(self.gateset.bulk_probs,evt,check=True)
      bulk_probs_new = self.assertNoWarnings(self.gateset._calc().bulk_probs_new,evt,check=True)
      self.assertArraysAlmostEqual(bulk_probs['plus'],bulk_probs_new['plus'])
      self.assertArraysAlmostEqual(bulk_probs['minus'],bulk_probs_new['minus'])

      nGateStrings = 2; nSpamLabels = 2
      probs_to_fill = np.empty( (nSpamLabels,nGateStrings), 'd')
      probs_to_fill_new = np.empty( (nSpamLabels,nGateStrings), 'd')
      spam_label_rows = { 'plus': 0, 'minus': 1 }
      self.assertNoWarnings(self.gateset.bulk_fill_probs, probs_to_fill, spam_label_rows, evt, check=True)
      self.assertNoWarnings(self.gateset._calc().bulk_fill_probs_new, probs_to_fill_new, spam_label_rows, evt, check=True)
      self.assertArraysAlmostEqual(probs_to_fill,probs_to_fill_new)


  def test_derivatives(self):
      gatestring0 = ('Gi','Gx')
      gatestring1 = ('Gx','Gy')
      gatestring2 = ('Gx','Gy','Gy')

      evt = self.gateset.bulk_evaltree( [gatestring0,gatestring1,gatestring2] )

      bulk_dP = self.gateset.bulk_dpr("plus", evt, returnPr=False, check=True)
      bulk_dP_m = self.gateset.bulk_dpr("minus", evt, returnPr=False, check=True)
      bulk_dP_chk, bulk_P = self.gateset.bulk_dpr("plus", evt, returnPr=True, check=False)
      bulk_dP_m_chk, bulk_Pm = self.gateset.bulk_dpr("minus", evt, returnPr=True, check=False)

      bulk_dP_new = self.gateset._calc().bulk_dpr_new("plus", evt, returnPr=False, check=True)
      bulk_dP_m_new = self.gateset._calc().bulk_dpr_new("minus", evt, returnPr=False, check=True)
      bulk_dP_chk_new, bulk_P_new = self.gateset._calc().bulk_dpr_new("plus", evt, returnPr=True, check=False)
      bulk_dP_m_chk_new, bulk_Pm_new = self.gateset._calc().bulk_dpr_new("minus", evt, returnPr=True, check=False)

      self.assertArraysAlmostEqual(bulk_dP,bulk_dP_new)
      self.assertArraysAlmostEqual(bulk_dP_m,bulk_dP_m_new)
      self.assertArraysAlmostEqual(bulk_dP_chk,bulk_dP_chk_new)
      self.assertArraysAlmostEqual(bulk_dP_m_chk,bulk_dP_m_chk_new)
      self.assertArraysAlmostEqual(bulk_P,bulk_P_new)
      self.assertArraysAlmostEqual(bulk_Pm,bulk_Pm_new)

      #Artificially reset the "smallness" threshold for scaling
      # to be sure to engate the scaling machinery
      PORIG = pygsti.objects.gscalc.PSMALL; pygsti.objects.gscalc.PSMALL = 10
      DORIG = pygsti.objects.gscalc.DSMALL; pygsti.objects.gscalc.DSMALL = 10
      bulk_dPb = self.gateset.bulk_dpr("plus", evt, returnPr=False, check=True)
      bulk_dPb_new = self.gateset._calc().bulk_dpr_new("plus", evt, returnPr=False, check=True)
      pygsti.objects.gscalc.PSMALL = PORIG
      bulk_dPc = self.gateset.bulk_dpr("plus", evt, returnPr=False, check=True)
      bulk_dPc_new = self.gateset._calc().bulk_dpr_new("plus", evt, returnPr=False, check=True)
      pygsti.objects.gscalc.DSMALL = DORIG
      self.assertArraysAlmostEqual(bulk_dPb,bulk_dPb_new)
      self.assertArraysAlmostEqual(bulk_dPc,bulk_dPc_new)


      bulk_dProbs = self.gateset.bulk_dprobs(evt, returnPr=False, check=True)
      bulk_dProbs_chk = self.gateset.bulk_dprobs(evt, returnPr=True, check=True)
      bulk_dProbs_new = self.gateset._calc().bulk_dprobs_new(evt, returnPr=False, check=True)
      bulk_dProbs_chk_new = self.gateset._calc().bulk_dprobs_new(evt, returnPr=True, check=True)

      self.assertArraysAlmostEqual(bulk_dProbs['plus'],bulk_dProbs_new['plus'])
      self.assertArraysAlmostEqual(bulk_dProbs_chk['plus'][0],
                                   bulk_dProbs_chk_new['plus'][0])
      self.assertArraysAlmostEqual(bulk_dProbs_chk['plus'][1],
                                   bulk_dProbs_chk_new['plus'][1])

      nGateStrings = 3; nSpamLabels = 2; nParams = self.gateset.num_params()
      probs_to_fill = np.empty( (nSpamLabels,nGateStrings), 'd')
      dprobs_to_fill = np.empty( (nSpamLabels,nGateStrings,nParams), 'd')
      dprobs_to_fillB = np.empty( (nSpamLabels,nGateStrings,nParams), 'd')

      probs_to_fill_new = np.empty( (nSpamLabels,nGateStrings), 'd')
      dprobs_to_fill_new = np.empty( (nSpamLabels,nGateStrings,nParams), 'd')
      dprobs_to_fillB_new = np.empty( (nSpamLabels,nGateStrings,nParams), 'd')

      spam_label_rows = { 'plus': 0, 'minus': 1 }
      self.assertNoWarnings(self.gateset.bulk_fill_dprobs, dprobs_to_fill, spam_label_rows, evt,
                            prMxToFill=probs_to_fill,check=True)
      self.assertNoWarnings(self.gateset._calc().bulk_fill_dprobs_new, dprobs_to_fill_new, spam_label_rows, evt,
                            prMxToFill=probs_to_fill_new,check=True)
      self.assertArraysAlmostEqual(dprobs_to_fill,dprobs_to_fill_new)
      self.assertArraysAlmostEqual(probs_to_fill,probs_to_fill_new)

      #without probs
      self.assertNoWarnings(self.gateset.bulk_fill_dprobs, dprobs_to_fillB, spam_label_rows, evt, check=True)
      self.assertNoWarnings(self.gateset._calc().bulk_fill_dprobs_new, dprobs_to_fillB_new, spam_label_rows, evt, check=True)
      self.assertArraysAlmostEqual(dprobs_to_fillB,dprobs_to_fillB_new)

      dProds = self.gateset.bulk_dproduct(evt)
      dProds_new = self.gateset._calc().bulk_dproduct_new(evt)
      self.assertArraysAlmostEqual(dProds, dProds_new)
      #with self.assertRaises(MemoryError):
      #    self.gateset._calc().bulk_dproduct_new(evt,memLimit=1)



  def test_hessians(self):
      gatestring0 = ('Gi','Gx')
      gatestring1 = ('Gx','Gy')
      gatestring2 = ('Gx','Gy','Gy')

      evt = self.gateset.bulk_evaltree( [gatestring0,gatestring1,gatestring2] )

      bulk_hP = self.gateset.bulk_hpr("plus", evt, returnPr=False, returnDeriv=False, check=True)
      bulk_hP_m = self.gateset.bulk_hpr("minus", evt, returnPr=False, returnDeriv=False, check=True)
      bulk_hP_chk, bulk_dP, bulk_P = self.gateset.bulk_hpr("plus", evt, returnPr=True, returnDeriv=True, check=False)
      bulk_hP_m_chk, bulk_dP_m, bulk_P_m = self.gateset.bulk_hpr("minus", evt, returnPr=True, returnDeriv=True, check=False)
      bulk_hP_new = self.gateset._calc().bulk_hpr_new("plus", evt, returnPr=False, returnDeriv=False, check=True)
      bulk_hP_m_new = self.gateset._calc().bulk_hpr_new("minus", evt, returnPr=False, returnDeriv=False, check=True)
      bulk_hP_chk_new, bulk_dP_new, bulk_P_new = self.gateset._calc().bulk_hpr_new("plus", evt, returnPr=True, returnDeriv=True, check=False)
      bulk_hP_m_chk_new, bulk_dP_m_new, bulk_P_m_new = self.gateset._calc().bulk_hpr_new("minus", evt, returnPr=True, returnDeriv=True, check=False)

      self.assertArraysAlmostEqual(bulk_hP,bulk_hP_new)
      self.assertArraysAlmostEqual(bulk_hP_m,bulk_hP_m_new)
      self.assertArraysAlmostEqual(bulk_hP_chk,bulk_hP_chk_new)
      self.assertArraysAlmostEqual(bulk_hP_m_chk,bulk_hP_m_chk_new)
      self.assertArraysAlmostEqual(bulk_dP,bulk_dP_new)
      self.assertArraysAlmostEqual(bulk_dP_m,bulk_dP_m_new)
      self.assertArraysAlmostEqual(bulk_P,bulk_P_new)
      self.assertArraysAlmostEqual(bulk_P_m,bulk_P_m_new)

      #Artificially reset the "smallness" threshold for scaling
      # to be sure to engate the scaling machinery
      PORIG = pygsti.objects.gscalc.PSMALL; pygsti.objects.gscalc.PSMALL = 10
      DORIG = pygsti.objects.gscalc.DSMALL; pygsti.objects.gscalc.DSMALL = 10
      HORIG = pygsti.objects.gscalc.HSMALL; pygsti.objects.gscalc.HSMALL = 10
      bulk_hPb = self.gateset.bulk_hpr("plus", evt, returnPr=False, returnDeriv=False, check=True)
      bulk_hPb_new = self.gateset._calc().bulk_hpr_new("plus", evt, returnPr=False, returnDeriv=False, check=True)
      pygsti.objects.gscalc.PSMALL = PORIG
      bulk_hPc = self.gateset.bulk_hpr("plus", evt, returnPr=False, returnDeriv=False, check=True)
      bulk_hPc_new = self.gateset._calc().bulk_hpr_new("plus", evt, returnPr=False, returnDeriv=False, check=True)
      pygsti.objects.gscalc.DSMALL = DORIG
      pygsti.objects.gscalc.HSMALL = HORIG
      self.assertArraysAlmostEqual(bulk_hPb,bulk_hPb_new)
      self.assertArraysAlmostEqual(bulk_hPc,bulk_hPc_new)


      bulk_hProbs = self.gateset.bulk_hprobs(evt, returnPr=False, check=True)
      bulk_hProbs_chk = self.gateset.bulk_hprobs(evt, returnPr=True, check=True)
      bulk_hProbs_new = self.gateset._calc().bulk_hprobs_new(evt, returnPr=False, check=True)
      bulk_hProbs_chk_new = self.gateset._calc().bulk_hprobs_new(evt, returnPr=True, check=True)
      self.assertArraysAlmostEqual(bulk_hProbs['plus'],bulk_hProbs_new['plus'])
      self.assertArraysAlmostEqual(bulk_hProbs['minus'],bulk_hProbs_new['minus'])
      self.assertArraysAlmostEqual(bulk_hProbs_chk['plus'][0],bulk_hProbs_chk_new['plus'][0])
      self.assertArraysAlmostEqual(bulk_hProbs_chk['minus'][0],bulk_hProbs_chk_new['minus'][0])
      self.assertArraysAlmostEqual(bulk_hProbs_chk['plus'][1],bulk_hProbs_chk_new['plus'][1])
      self.assertArraysAlmostEqual(bulk_hProbs_chk['minus'][1],bulk_hProbs_chk_new['minus'][1])

      #Vary keyword args
      bulk_hProbs_B = self.gateset.bulk_hprobs(evt, returnPr=True, returnDeriv=True)
      bulk_hProbs_C = self.gateset.bulk_hprobs(evt, returnDeriv=True)
      bulk_hProbs_B_new = self.gateset._calc().bulk_hprobs_new(evt, returnPr=True, returnDeriv=True)
      bulk_hProbs_C_new = self.gateset._calc().bulk_hprobs_new(evt, returnDeriv=True)
      for sl in ['plus','minus']:
          for i in range(len(bulk_hProbs_B[sl])):
              self.assertArraysAlmostEqual(bulk_hProbs_B[sl][i],bulk_hProbs_B_new[sl][i])
          for i in range(len(bulk_hProbs_C[sl])):
              self.assertArraysAlmostEqual(bulk_hProbs_C[sl][i],bulk_hProbs_C_new[sl][i])


      nGateStrings = 3; nSpamLabels = 2; nParams = self.gateset.num_params()
      probs_to_fill = np.empty( (nSpamLabels,nGateStrings), 'd')
      probs_to_fillB = np.empty( (nSpamLabels,nGateStrings), 'd')
      dprobs_to_fill = np.empty( (nSpamLabels,nGateStrings,nParams), 'd')
      dprobs_to_fillB = np.empty( (nSpamLabels,nGateStrings,nParams), 'd')
      hprobs_to_fill = np.empty( (nSpamLabels,nGateStrings,nParams,nParams), 'd')
      hprobs_to_fillB = np.empty( (nSpamLabels,nGateStrings,nParams,nParams), 'd')

      probs_to_fill_new = np.empty( (nSpamLabels,nGateStrings), 'd')
      probs_to_fillB_new = np.empty( (nSpamLabels,nGateStrings), 'd')
      dprobs_to_fill_new = np.empty( (nSpamLabels,nGateStrings,nParams), 'd')
      dprobs_to_fillB_new = np.empty( (nSpamLabels,nGateStrings,nParams), 'd')
      hprobs_to_fill_new = np.empty( (nSpamLabels,nGateStrings,nParams,nParams), 'd')
      hprobs_to_fillB_new = np.empty( (nSpamLabels,nGateStrings,nParams,nParams), 'd')

      spam_label_rows = { 'plus': 0, 'minus': 1 }
      self.assertNoWarnings(self.gateset.bulk_fill_hprobs, hprobs_to_fill, spam_label_rows, evt,
                            prMxToFill=probs_to_fill, derivMxToFill=dprobs_to_fill, check=True)
      self.assertNoWarnings(self.gateset._calc().bulk_fill_hprobs_new, hprobs_to_fill_new, spam_label_rows, evt,
                            prMxToFill=probs_to_fill_new, derivMxToFill=dprobs_to_fill_new, check=True)

      self.assertArraysAlmostEqual(hprobs_to_fill,hprobs_to_fill_new)
      self.assertArraysAlmostEqual(dprobs_to_fill,dprobs_to_fill_new)
      self.assertArraysAlmostEqual(probs_to_fill ,probs_to_fill_new)

      #without derivative
      self.assertNoWarnings(self.gateset.bulk_fill_hprobs, hprobs_to_fillB, spam_label_rows, evt,
                            prMxToFill=probs_to_fillB, check=True)
      self.assertNoWarnings(self.gateset._calc().bulk_fill_hprobs_new, hprobs_to_fillB_new, spam_label_rows, evt,
                            prMxToFill=probs_to_fillB_new, check=True) 
      self.assertArraysAlmostEqual(hprobs_to_fillB,hprobs_to_fillB_new)
      self.assertArraysAlmostEqual(probs_to_fillB,probs_to_fillB_new)

      #without probs
      self.assertNoWarnings(self.gateset.bulk_fill_hprobs, hprobs_to_fillB, spam_label_rows, evt,
                            derivMxToFill=dprobs_to_fillB, check=True) 
      self.assertNoWarnings(self.gateset._calc().bulk_fill_hprobs_new, hprobs_to_fillB_new, spam_label_rows, evt,
                            derivMxToFill=dprobs_to_fillB_new, check=True) 
      self.assertArraysAlmostEqual(hprobs_to_fillB,hprobs_to_fillB_new)
      self.assertArraysAlmostEqual(dprobs_to_fillB,dprobs_to_fillB_new)

      #without either
      self.assertNoWarnings(self.gateset.bulk_fill_hprobs, hprobs_to_fillB, spam_label_rows, evt, check=True) 
      self.assertNoWarnings(self.gateset._calc().bulk_fill_hprobs_new, hprobs_to_fillB_new, spam_label_rows, evt, check=True) 
      self.assertArraysAlmostEqual(hprobs_to_fillB,hprobs_to_fillB_new)

      N = self.gateset.get_dimension()**2 #number of elements in a gate matrix
      
      hProds = self.gateset.bulk_hproduct(evt)
      hProdsB,scales = self.gateset.bulk_hproduct(evt, bScale=True)
      hProds_new = self.gateset._calc().bulk_hproduct_new(evt)
      hProdsB_new,scales_new = self.gateset._calc().bulk_hproduct_new(evt, bScale=True)
      self.assertArraysAlmostEqual(hProds, hProds_new)
      self.assertArraysAlmostEqual(scales[:,None,None,None,None]*hProdsB,
                                   scales_new[:,None,None,None,None]*hProdsB_new)

      hProdsFlat = self.gateset.bulk_hproduct(evt, flat=True, bScale=False)
      hProdsFlatB,S1 = self.gateset.bulk_hproduct(evt, flat=True, bScale=True)
      hProdsFlat_new = self.gateset._calc().bulk_hproduct_new(evt, flat=True, bScale=False)
      hProdsFlatB_new,S1_new = self.gateset._calc().bulk_hproduct_new(evt, flat=True, bScale=True)

      self.assertArraysAlmostEqual(hProdsFlat, hProdsFlat_new)
      self.assertArraysAlmostEqual(np.repeat(S1,N)[:,None,None]*hProdsFlatB, np.repeat(S1_new,N)[:,None,None]*hProdsFlatB_new)

      hProdsC, dProdsC, prodsC = self.gateset.bulk_hproduct(evt, bReturnDProdsAndProds=True, bScale=False)
      hProdsD, dProdsD, prodsD, S2 = self.gateset.bulk_hproduct(evt, bReturnDProdsAndProds=True, bScale=True)
      hProdsC_new, dProdsC_new, prodsC_new = self.gateset._calc().bulk_hproduct_new(evt, bReturnDProdsAndProds=True, bScale=False)
      hProdsD_new, dProdsD_new, prodsD_new, S2_new = self.gateset._calc().bulk_hproduct_new(evt, bReturnDProdsAndProds=True, bScale=True)

      self.assertArraysAlmostEqual(hProdsC, hProdsC_new)
      self.assertArraysAlmostEqual(dProdsC, dProdsC_new)
      self.assertArraysAlmostEqual(prodsC, prodsC_new)
      self.assertArraysAlmostEqual(S2[:,None,None,None,None]*hProdsD, S2_new[:,None,None,None,None]*hProdsD_new)
      self.assertArraysAlmostEqual(S2[:,None,None,None]*dProdsD, S2_new[:,None,None,None]*dProdsD_new)
      self.assertArraysAlmostEqual(S2[:,None,None]*prodsD, S2_new[:,None,None]*prodsD_new)

      hProdsF, dProdsF, prodsF    = self.gateset.bulk_hproduct(evt, bReturnDProdsAndProds=True, flat=True, bScale=False)
      hProdsF2, dProdsF2, prodsF2, S3 = self.gateset.bulk_hproduct(evt, bReturnDProdsAndProds=True, flat=True, bScale=True)
      hProdsF_new, dProdsF_new, prodsF_new    = self.gateset.bulk_hproduct(evt, bReturnDProdsAndProds=True, flat=True, bScale=False)
      hProdsF2_new, dProdsF2_new, prodsF2_new, S3_new = self.gateset._calc().bulk_hproduct_new(evt, bReturnDProdsAndProds=True, flat=True, bScale=True)
      self.assertArraysAlmostEqual(hProdsF, hProdsF_new)
      self.assertArraysAlmostEqual(dProdsF, dProdsF_new)
      self.assertArraysAlmostEqual(prodsF, prodsF_new)
      self.assertArraysAlmostEqual(np.repeat(S3,N)[:,None,None]*hProdsF2, np.repeat(S3_new,N)[:,None,None]*hProdsF2_new)
      self.assertArraysAlmostEqual(np.repeat(S3,N)[:,None]*dProdsF2, np.repeat(S3_new,N)[:,None]*dProdsF2_new)
      self.assertArraysAlmostEqual(S3[:,None,None]*prodsF2, S3_new[:,None,None]*prodsF2_new)



      
if __name__ == "__main__":
    unittest.main(verbosity=2)
