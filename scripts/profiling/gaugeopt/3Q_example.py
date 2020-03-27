
# coding: utf-8

# In[1]:

import pygsti
import pickle
import time


# In[2]:

gs_target = pygsti.construction.build_gateset(
[8], [('Q0','Q1','Q2')],['Gx1','Gy1','Gx2','Gy2','Gx3','Gy3','Gcnot12','Gcnot23'],
[ "X(pi/2,Q0):I(Q1):I(Q2)", "Y(pi/2,Q0):I(Q1):I(Q2)", "I(Q0):X(pi/2,Q1):I(Q2)", "I(Q0):Y(pi/2,Q1):I(Q2)",
  "I(Q0):I(Q1):X(pi/2,Q2)", "I(Q0):I(Q1):Y(pi/2,Q2)", "CX(pi,Q0,Q1):I(Q2)", "I(Q0):CX(pi,Q1,Q2)"],
prep_labels=['rho0'], prep_expressions=["0"],
effect_labels=['E0','E1','E2','E3','E4','E5','E6'], effect_expressions=["0","1","2","3","4","5","6"],
spamdefs={'upupup': ('rho0','E0'), 'upupdn': ('rho0','E1'), 'updnup': ('rho0','E2'), 'updndn': ('rho0','E3'),
'dnupup': ('rho0','E4'), 'dnupdn': ('rho0','E5'), 'dndnup': ('rho0','E6'), 'dndndn': ('rho0','remainder')},
basis="pp")
#print gs_target.num_params()


# In[3]:

#Test Gauge optimization
gs_depol = gs_target.copy().depolarize(max_gate_noise=0.05, spam_noise=0.1, seed=1200)
gs_kicked = gs_depol.kick(absmag=0.25, seed=1200)


# In[ ]:

t0 = time.time()
gs_go = pygsti.gaugeopt_to_target(gs_kicked, gs_target, tol=1e-10, verbosity = 3)
print("%g sec" % (time.time()-t0))
print(gs_go.frobeniusdist(gs_target))


# In[ ]:

pickle.dump(results, open("3qbit_results.pkl", "wb"))


# In[ ]:



