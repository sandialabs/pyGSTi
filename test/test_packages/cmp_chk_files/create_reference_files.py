import pygsti
from pygsti.modelpacks.legacy import std1Q_XYI
import pickle
import os

pv = pygsti.__version__
if len(pv.split('.')) > 3:
    pv = '.'.join(pv.split('.')[0:3])
print("PyGSTi version ", pv)

target_model = std1Q_XYI.target_model()

# 2) get the building blocks needed to specify which operation sequences are needed
prep_fiducials, meas_fiducials = std1Q_XYI.prepStrs, std1Q_XYI.effectStrs
germs = std1Q_XYI.germs
maxLengths = [1, 2, 4]  # roughly gives the length of the sequences used by GST

# 3) generate "fake" data from a depolarized version of target_model
mdl_datagen = target_model.depolarize(op_noise=0.01, spam_noise=0.001)
listOfExperiments = pygsti.construction.make_lsgst_experiment_list(
    target_model, prep_fiducials, meas_fiducials, germs, maxLengths)
ds = pygsti.construction.generate_fake_data(mdl_datagen, listOfExperiments, n_samples=1000,
                                            sample_error="binomial", seed=1234)

results = pygsti.do_stdpractice_gst(ds, target_model, prep_fiducials, meas_fiducials,
                                    germs, maxLengths, verbosity=3)


def outname(typ):
    nm = "pygsti" + pv + "." + typ
    if os.path.exists(nm):
        raise ValueError("File %s already exists! Will not overwrite it - you must remove it first." % nm)
    print("Writing ", nm)
    return nm


# Dataset object
pygsti.io.write_dataset(outname("dataset.txt"), ds)  # text dataset
ds.save(outname("dataset"))  # binary dataset
with open(outname("dataset.pkl"), "wb") as f:  # pickled dataset
    pickle.dump(ds, f)

# Model object
pygsti.io.write_model(mdl_datagen, outname("gateset.txt"))  # text model
with open(outname("gateset.pkl"), "wb") as f:  # pickled model
    pickle.dump(mdl_datagen, f)

# Results object
with open(outname("results.pkl"), "wb") as f:  # pickled results
    pickle.dump(results, f)
