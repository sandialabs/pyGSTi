#!/usr/bin/env python3
import time
import pickle
import pygsti
from pygsti.construction import std2Q_XYCNOT

gs_target = pygsti.construction.build_gateset( 
            [4], [('Q0','Q1')],['Gix','Giy','Gxi','Gyi','Gcnot'], 
            [ "X(pi/2,Q1)", "Y(pi/2,Q1)", "X(pi/2,Q0)", "Y(pi/2,Q0)", "CNOT(Q0,Q1)" ],
            prep_labels=['rho0'], prep_expressions=["0"],
            effect_labels=['E0','E1','E2'], effect_expressions=["0","1","2"], 
            spamdefs={'upup': ('rho0','E0'), 'updn': ('rho0','E1'),
                      'dnup': ('rho0','E2'), 'dndn': ('rho0','remainder') }, basis="pp")

gs_targetB = pygsti.construction.build_gateset( 
            [4], [('Q0','Q1')],['Gix','Giy','Gxi','Gyi','Gcnot'], 
            [ "I(Q0):X(pi/2,Q1)", "I(Q0):Y(pi/2,Q1)", "X(pi/2,Q0):I(Q1)", "Y(pi/2,Q0):I(Q1)", "CNOT(Q0,Q1)" ],
            prep_labels=['rho0'], prep_expressions=["0"], 
            effect_labels=['E0','E1','E2'], effect_expressions=["0","1","2"], 
            spamdefs={'upup': ('rho0','E0'), 'updn': ('rho0','E1'),
                      'dnup': ('rho0','E2'), 'dndn': ('rho0','remainder') }, basis="pp")

# If you're lucky and your gateset is one of pyGSTi's "standard" gate sets, you can just import it.
gs_targetC = std2Q_XYCNOT.gs_target

#check that these are all the same
assert(abs(gs_target.frobeniusdist(gs_targetB)) < 1e-6)
assert(abs(gs_target.frobeniusdist(gs_targetC)) < 1e-6)


# ### Step 2: Obtain lists of fiducial and germ gate sequences
# These are the building blocks of the gate sequences performed in the experiment. Typically, these lists are either given to you by the folks at Sandia National Labs (email pygsti@sandia.gov), provided by pyGSTi because you're using a "standard" gate set, or computed using "fiducial selection" and "germ selection" algorithms (which are a part of pyGSTi, but not covered in this tutorial).

# In[4]:

#If you know the fiducial strings you can create a list manually.  Note
# that in general there can be different "preparation" and "measurement"
# (or "effect") fiducials.
prep_fiducials = pygsti.construction.gatestring_list( [ (), ('Gix',), ('Giy',), ('Gix','Gix'), 
('Gxi',), ('Gxi','Gix'), ('Gxi','Giy'), ('Gxi','Gix','Gix'), 
('Gyi',), ('Gyi','Gix'), ('Gyi','Giy'), ('Gyi','Gix','Gix'), 
('Gxi','Gxi'), ('Gxi','Gxi','Gix'), ('Gxi','Gxi','Giy'), ('Gxi','Gxi','Gix','Gix') ] )

effect_fiducials = pygsti.construction.gatestring_list( [(), ('Gix',), ('Giy',), 
 ('Gix','Gix'), ('Gxi',),
 ('Gyi',), ('Gxi','Gxi'),
 ('Gxi','Gix'), ('Gxi','Giy'),
 ('Gyi','Gix'), ('Gyi','Giy')] )

#Or, if you're lucky, you can just import them
prep_fiducialsB = std2Q_XYCNOT.prepStrs
effect_fiducialsB = std2Q_XYCNOT.effectStrs

#check that these are the same
assert(prep_fiducials == prep_fiducialsB)
assert(effect_fiducials == effect_fiducialsB)

#Use fiducial sequences to create a "spam specifiers" object, telling
# GST which preparation and measurement fiducials to follow and precede which
# state preparation and effect operators, respectively.
specs = pygsti.construction.build_spam_specs(
    prep_strs=prep_fiducials,
    effect_strs=effect_fiducials,
    prep_labels=gs_target.get_prep_labels(),
    effect_labels=gs_target.get_effect_labels() )

#Alternatively, if you're lucky, you can grab the specs directly:
specsB = std2Q_XYCNOT.specs
assert(specs[0] == specsB[0])


# In[5]:

#germ lists can be specified in the same way.  In this case, there are
# 71 germs required to do honest GST.  Since this would crowd this tutorial
# notebook, we create some smaller lists of germs manually and import the
# full 71-germ list from std2Q_XYCNOT
germs4 = pygsti.construction.gatestring_list(
    [ ('Gix',), ('Giy',), ('Gxi',), ('Gyi',) ] )
germs11 = pygsti.construction.gatestring_list( [ ('Gix',), ('Giy',), ('Gxi',), ('Gyi',), ('Gcnot',), ('Gxi','Gyi'), ('Gix','Giy'), ('Gix','Gcnot'), ('Gxi','Gcnot'), ('Giy','Gcnot'), ('Gyi','Gcnot') ] )

germs71 = std2Q_XYCNOT.germs

#A list of maximum lengths for each GST iteration
maxLengths = [1,2,4]

#Create a list of GST experiments for this gateset, with
#the specified fiducials, germs, and maximum lengths.  We use
#"germs4" here so that the tutorial runs quickly; really, you'd
#want to use germs71!
listOfExperiments = pygsti.construction.make_lsgst_experiment_list(gs_target.gates.keys(), prep_fiducials,
                                                                   effect_fiducials, germs4, maxLengths)

#Create an empty dataset file, which stores the list of experiments
# and zerod-out columns where data should be inserted.  Note the use of the SPAM
# labels in the "Columns" header line.
pygsti.io.write_empty_dataset("tutorial_files/My2QDataTemplate.txt", listOfExperiments,
                              "## Columns = upup count, updn count, dnup count, dndn count")

#Generate some "fake" (simulated) data based on a depolarized version of the target gateset
gs_datagen = gs_target.depolarize(gate_noise=0.1, spam_noise=0.001)
ds = pygsti.construction.generate_fake_data(gs_datagen, listOfExperiments, n_samples=1000,
                                            sample_error="multinomial", seed=2016)


start = time.time()
'''
results = pygsti.do_long_sequence_gst(ds, gs_target, prep_fiducials, effect_fiducials, germs4,
                                    maxLengths, gaugeOptParams={'item_weights': {'spam':0.1,'gates': 1.0}},
                                    advancedOptions={ 'depolarizeStart' : 0.1 }, mem_limit=3*(1024)**3,
                                    verbosity=3 )
'''
results = pygsti.do_long_sequence_gst(ds, gs_target, prep_fiducials, effect_fiducials, germs4,
                                    maxLengths, gauge_opt_params=None,
                                    advanced_options={ 'depolarizeStart' : 0.1 }, mem_limit=3*(1024)**3,
                                    verbosity=3 )
end = time.time()
print("Total time=%f hours" % ((end - start) / 3600.0))

#If you wanted to, you could pickle the results for later analysis:
pickle.dump(results, open("gaugeopt/2qbit_results.pkl", "wb"))
