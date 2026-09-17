---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# File Input and Output

This tutorial describes how different pyGSTi objects can be saved to and read back from files.  There are four  main ways of doing this, depending on the type of object you're wanting to save.

1. **Custom text formats** A small number of pyGSTi objects can be converted to and from a pyGSTi-specific text format, that can be written to and from files.  Currently `DataSet`, `MultiDataSet`, `Circuit`, and `Label` are the only such objects.  `Circuit` and `Label` objects are usually not read or written individually; usually it's a *list* of circuits that are.  

2. **JSON format** Many of pyGSTi's objects have the ability to be converted to and from the JavaScript object notation (JSON) format.  This is a standard text format that is widely supported and portable, making it the preferred choice for reading and writing just about any pyGSTi object except a data set or and circuit list (which have their own text formats).

3. **Experiment design directory** Several high-level objects can be read and written from *directories* rather than individual files.  These include `ExperimentDesign`, `Protocol`, `ProtocolData`, `ProtocolResults`, and `ProtocolResultsDir` objects.  Because these object types hold lots of other objects, storing them to directories makes looking through the contents of the object possible without even using Python, since different parts of the object are stored in different files.  This is further facilitated by saving the parts of the high-level object in the custom text or JSON formats described above - the resulting directory contains portable, human-readable files.  
   Secondly, this directory format allows for multiple of object types to effectively *share* a single directory. This is achieved by placing the data for an object in a subdirectory beneath the given directory named based on the object type.  For instance, writing an `ExperimentDesign` to the *path/to/root_directory* will write the object's contents within the *path/to/root_directory/edesign* directory.  A `ProtocolData`, which contains an `ExperimentDesign`, when writing to the same *path/to/root_directory* will write its non-experiment design contents within *path/to/root_directory/data* directory and re-use the serialized experiment design in the `edesign` sub-directory.  When experiment designs contain other (nested) experiment designs, this leads to an efficient tree structure organizing circuit lists, experimental data, and analysis results.   

4. **Pickle, basic JSON and MSGPACK formats**  Finally, most pyGSTi objects are able to be "pickled" using Python's pickle format.  The codec that allows for this can also write [JSON](https://www.json.org) and [MessagePack](https://msgpack.org/index.html) formats via functions in `pygsti.io.json` and `pygsti.io.msgpack`.  These routines create an alternate format to the "nice" JSON format described above: they produce a format that is fragile, non-portable, and not human readable.  As such, using `pickle`, `pygsti.io.json` or `pygsti.io.msgpack` should be a last resort and only used for temporary storage.


## Text formats

All text-based input and output is done via the `pygsti.io` sub-package.  Below we give examples of how to read and write data sets and circuit objects.  (Note: Data set objects also have `read_binary` and `write_binary` methods which read and write *binary* formats -- different from the text formats used by the `pygsti.io` routines -- but these binary formats are less portable and not human readable, making the text formats preferable).

```{code-cell} ipython3
import pygsti

#DataSets ------------------------------------------------------------
dataset_txt = \
"""## Columns = 0 count, 1 count
{} 0 100
Gx 10 80
GxGy 40 20
Gx^4 20 70
"""
with open("../../tutorial_files/TestDataSet.txt","w") as f:
    f.write(dataset_txt)

ds = pygsti.io.read_dataset("../../tutorial_files/TestDataSet.txt")
pygsti.io.write_dataset("../../tutorial_files/TestDataSet.txt", ds)


#MultiDataSets ------------------------------------------------------------
multidataset_txt = \
"""## Columns = DS0 0 count, DS0 1 count, DS1 0 frequency, DS1 count total                                
{} 0 100 0 100                                                                                                      
Gx 10 90 0.1 100                                                                                                    
GxGy 40 60 0.4 100                                                                                                  
Gx^4 20 80 0.2 100                                                                                                  
"""

with open("../../tutorial_files/TestMultiDataSet.txt","w") as f:
    f.write(multidataset_txt)
    
multiDS = pygsti.io.read_multidataset("../../tutorial_files/TestMultiDataSet.txt", cache=True)
pygsti.io.write_multidataset("../../tutorial_files/TestDataSet.txt", multiDS)


#DataSets w/timestamped data --------------------------------------------
# Note: left of equals sign is letter, right is spam label
tddataset_txt = \
"""## 0 = 0
## 1 = 1
{} 011001
Gx 111000111
Gy 11001100
"""
with open("../../tutorial_files/TestTDDataset.txt","w") as f:
    f.write(tddataset_txt)
    
tdds_fromfile = pygsti.io.read_time_dependent_dataset("../../tutorial_files/TestTDDataset.txt")
#NOTE: currently there's no way to *write* a DataSet w/timestamped data to a text file yet.


#Circuits ------------------------------------------------------------
from pygsti.modelpacks import smq1Q_XY  
cList = pygsti.circuits.create_lsgst_circuits(
    [('Gxpi2',0), ('Gypi2',0)], smq1Q_XY.prep_fiducials(), smq1Q_XY.meas_fiducials(),
    smq1Q_XY.germs(), [1,2,4,8])    
pygsti.io.write_circuit_list("../../tutorial_files/TestCircuitList.txt",cList,"#Test Circuit List")
pygsti.io.write_empty_dataset("../../tutorial_files/TestEmptyDataset.txt",cList) 
  #additionally creates columns of zeros where data should go...
cList2 = pygsti.io.read_circuit_list("../../tutorial_files/TestCircuitList.txt")
```

## JSON format
Any object that derives from `pygsti.baseobjs.NicelySerializable` can be written to or read from a "nice" JSON format using the object's `write` and `read` methods.  This is the best way to serialize `Model` objects, which can be particularly complex.  We give an example of this below.

```{code-cell} ipython3
pspec = pygsti.processors.QubitProcessorSpec(2, ['Gxpi2', 'Gypi2', 'Gcnot'], geometry='line')
mdl = pygsti.models.modelconstruction.create_crosstalk_free_model(pspec)
mdl.write("../../tutorial_files/serialized_model.json")

#A base class can read in derived-class files 
loaded_mdl = pygsti.models.Model.read("../../tutorial_files/serialized_model.json")
print(type(loaded_mdl))

import numpy as np
assert(np.allclose(mdl.to_vector(), loaded_mdl.to_vector()))
```

We can also serialize a `QubitProcessorSpec` this way:

```{code-cell} ipython3
pspec.write("../../tutorial_files/serialized_pspec.json")
print("wrote a", str(type(pspec)))
```

But we'll get an error if we try to read in the processor specification using `pygsti.models.Model`, since a class's `.read` ensures that it constructs an object of that type or a subclass of it:

```{code-cell} ipython3
try:
    pygsti.models.Model.read("../../tutorial_files/serialized_pspec.json")
except ValueError as v:
    print("ERROR: ", str(v))

p1 = pygsti.processors.QubitProcessorSpec.read("../../tutorial_files/serialized_pspec.json")  # ok - reads object of same type
p2 = pygsti.processors.ProcessorSpec.read("../../tutorial_files/serialized_pspec.json")  # ok - ProcessorSpec is a base class of QubitProcessorSpec
p3 = pygsti.baseobjs.NicelySerializable.read("../../tutorial_files/serialized_pspec.json") # also ok - NicelySerializeable is a base class too

assert(type(p1) == pygsti.processors.QubitProcessorSpec)
assert(type(p2) == pygsti.processors.QubitProcessorSpec)
assert(type(p3) == pygsti.processors.QubitProcessorSpec)
```

## Experiment design directory tree
`ExperimentDesign`, `Protocol`, `ProtocolData`, `ProtocolResults`, and `ProtocolResultsDir` objects can be written to a *directory* using using their `write` methods.  These objects place their data in subdirectories beneath the given directory as follows:

- `ExperimentDesign` places its contents in an `edesign` subdirectory.
- `ProtocolData` places its contained experiment design in an `edesign` subdirectory (it writes its contained `ExperimentDesign` to the same root directory) and the rest of its contents in a `data` subdirectory.
- `ProtocolResults` writes its contained `ProtocolData` object into the same root directory (this uses `edesign` and `data` subdirectories) and the rest of its contents into the `results/<name>` subdirectory, where `<name>` is given by the `name` attribute of the `ProtocolResults` object. 
- `ProtocolResultsDir` writes its contained `ProtocolData` object into the same root directory (this uses `edesign` and `data` subdirectories) and the rest of its contents, which consist of multiple `ProtocolResults` objects beneath a `results` subdirectory (i.e. it writes all of its contained `ProtocolResults` objects to the same root directory).
- `Protocol` doesn't conform to the same structure, and can be independently written to a given directory. Contents are written directly into this directory (no subdirectories), and so the destination directory is unrelated to the common "root directory" used by the preceding classes.

This results in a directory structure where common information (e.g. an experiment design) is only written to disk in one location (the `edesign` subdirectory) and shared by all the objects that need it.  Additional subdirectories may exist beneath the root directory to support *nested* experiment designs and their related data and results.  These subdirectory names are listed in the `edesign/subdirs.json` file.  Additional subdirectories not listed here are ignored and can be used by users as they please.  To read these objects back from disk, you use the following `pygsti.io` functions:

- `pygsti.io.read_edesign_from_dir(root_directory)` reads the experiment design saved beneath the given root directory, i.e., from `root_directory/edesign`.  Also loads nested experiment designs.
- `pygsti.io.read_data_from_dir(root_directory)` reads a `ProtocolData` from the given root directory, including nested data objects.
- `pygsti.io.read_results_from_dir(root_directory)` reads a `ProtocolResultsDir` from the given root directory, including nested result directory objects.
- `pygsti.io.read_protocol_from_dir(directory)` reads a `Protocol` object from the given directory.  This directory is independent and only used for holding a protocol's contents, i.e. *not* like the `root_directory` arguments to the previous functions.

We demonstrate some of this functionality with a simple example:

```{code-cell} ipython3
try:
    import shutil
    shutil.rmtree("../../tutorial_files/example_root_directory")  # start with a clean slate, in case you rerun this tutorial
except FileNotFoundError:
    pass
from pygsti.modelpacks import smq1Q_XYI
edesign = smq1Q_XYI.create_gst_experiment_design(1)
edesign.write("../../tutorial_files/example_root_directory")  # creates .../example_root_directory/edesign
```

```{code-cell} ipython3
# load back in the experiment design (just to demo this function):
loaded_edesign = pygsti.io.read_edesign_from_dir("../../tutorial_files/example_root_directory")

#Create & write an empty ProtocolData object to the same root directory
# (this is often useful because it creates an empty data set file in .../example_root_directory/data
#  that can be used as a template for the real experimental data)
pygsti.io.write_empty_protocol_data("../../tutorial_files/example_root_directory", loaded_edesign, clobber_ok=True)

#Alternatively, we can create an empty ProtocolData object and then write it:
data = pygsti.protocols.ProtocolData(loaded_edesign)
data.write()  # same as .write("../../tutorial_files/example_root_directory") because loaded_edesign remembers 
              # where is was loaded from and this is the default destination (see loaded_edesign._loaded_from)
```

```{code-cell} ipython3
#Next, let's create a protocol to run on the data.
protocol = pygsti.protocols.GateSetTomography(smq1Q_XYI.target_model("full TP"), name="TP_GST_Analysis")

#We can read/write this separately, though we don't really need to since it will get saved as a part
# of the results later on.  Still, to demonstrate this, here's how to do it:
protocol.write("../../tutorial_files/example_gst_protocol")
loaded_protocol = pygsti.io.read_protocol_from_dir("../../tutorial_files/example_gst_protocol")
```

```{code-cell} ipython3
#Now, generate some simulated data and run the protocol on it to produce some results

# Mimic filling in or replacing the empty "template" dataset file (dataset.txt) with actual data
datagen_model = smq1Q_XYI.target_model().depolarize(op_noise=0.05, spam_noise=0.2)
pygsti.io.fill_in_empty_dataset_with_fake_data("../../tutorial_files/example_root_directory/data/dataset.txt",
                                               datagen_model, num_samples=1000, seed=2021)

# Load the just-simulated data as a ProtocolData object and run our GST protocol on it:
loaded_data = pygsti.io.read_data_from_dir("../../tutorial_files/example_root_directory")
results = protocol.run(loaded_data)
```

```{code-cell} ipython3
# Save the results:
results.write()  # same as .write("../../tutorial_files/example_root_directory") because result's edesign remembers 
                 # the root directory is was loaded from and this is the default destination.
```

At this point we have directory containing an experiment design, a corresponding set of data, and some results from analyzing that data.  It may be interesting to poke around in this [example_root_directory](../../tutorial_files/example_root_directory) and take note of where things are.  Notice how the subdirectory under `results` is the name we gave to the protocol above: `TP_GST_Analysis`.

We can load the results back in, again using the *same* root directory:

```{code-cell} ipython3
loaded_results_dir = pygsti.io.read_results_from_dir("../../tutorial_files/example_root_directory")

print("Available results = ", list(loaded_results_dir.for_protocol.keys()))
gst_results = loaded_results_dir.for_protocol['TP_GST_Analysis']

gst_estimate = gst_results.estimates['TP_GST_Analysis']  # protocol name is also used as a default estimate name
gst_model = gst_estimate.models['stdgaugeopt']
dataset = gst_results.data.dataset  # ProtocolResults -> ProtocolData -> DataSet

nSigma = pygsti.tools.two_delta_logl_nsigma(gst_model, dataset)
print("Model violation (goodness of fit) = %g sigma" % nSigma)
if nSigma < 2:
    print("  The GST model gives a very good fit to the data!")
```

Something we want to emphasize is that all of the I/O function calls (except the simulated data generation and unnecessary separate saving of the protocol) are given the same, single root directory (`"../../tutorial_files/example_root_directory"`).  This is a key feature and the intended design of pyGSTi's I/O framework - that a given experiment design, data, and analysis all live together in the same "node" (corresponding to a root directory) in a potential tree of experiments.  In the example above, we just have a single node.  By using nested experiment designs (e.g. `CombinedExperimentDesign` and `SimultaneousExperimentDesign` objects) a tree of such "root directories" (perhaps better called "node directories" then) can be built.
