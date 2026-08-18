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

# Data sets

A `DataSet` holds experimental or simulated data as outcome counts. For time-independent data (the typical case, and all this page covers) it behaves much like a nested dictionary keyed first by circuit and then by outcome label, so `dataset[circuit][outcome_label]` reads and writes the number of times `outcome_label` was observed when running `circuit`.

Two things separate a `DataSet` from an actual dictionary of dictionaries.

First, a `DataSet` is either *non-static* or *static*. Non-static is the data-entry mode: you can freely add and modify counts. Calling `done_adding_data` switches the set to static, which restructures its internal storage for the access patterns pyGSTi's algorithms use and makes it read-only for the rest of its life. To change a static set's data, make a non-static copy with `copy_nonstatic` and modify that.

Second, because a `DataSet` can also carry time-dependent data, the single-outcome access syntax `dataset[circuit][outcome_label]` cannot create a *new* circuit key. Use the `add_`*xxx* methods for that. Once the circuit exists, bracket assignment works fine for changing its counts.

A finished, static `DataSet` is what you hand to pyGSTi's algorithm and driver routines to get a `Model` estimate. This page is about building one and looking at what's inside it.

```{code-cell} ipython3
import pygsti
```

## Creating a data set

There are three basic routes.

* Build an empty `DataSet` and add counts yourself. Remember that `add_`*xxx* is required for circuits not yet present, and call `done_adding_data` when you're finished.
* Load a text-format data file with `pygsti.io.read_dataset`. The result is already static, so no `done_adding_data` call is needed.
* Generate fake data from a `Model` with `pygsti.data.simulate_data`. This is how you simulate GST runs and compare them against your experimental results.

Each of the next three cells does one of these.

```{code-cell} ipython3
#1) Creating a data set from scratch
#    Note that tuples may be used in lieu of Circuit objects
ds1 = pygsti.data.DataSet(outcome_labels=['0','1'])
ds1.add_count_dict( ('Gx',), {'0': 10, '1': 90} )
ds1.add_count_dict( ('Gx','Gy'), {'0': 40, '1': 60} )
ds1[('Gy',)] = {'0': 10, '1': 90} # dictionary assignment

#Modify existing data using dictionary-like access
ds1[('Gx',)]['0'] = 15
ds1[('Gx',)]['1'] = 85

#Circuit objects can be used.
c = pygsti.circuits.Circuit( ('Gx','Gy'))
ds1[c]['0'] = 45
ds1[c]['1'] = 55

ds1.done_adding_data()
```

```{code-cell} ipython3
#2) By creating and loading a text-format dataset file.  The first
#    row is a directive which specifies what the columns (after the
#    first one) hold.  Note that "0" and "1" here are the
#    outcome labels and must match those of any Model used in
#    conjuction with this DataSet.
dataset_txt = \
"""## Columns = 0 count, 1 count
{}@(0)             0 100
Gxpi2:0@(0)        10 90
Gxpi2:0Gypi2:0@(0) 40 60
Gxpi2:0^4@(0)      20 80
"""
with open("../../../tutorial_files/Example_TinyDataset.txt","w") as tinydataset:
    tinydataset.write(dataset_txt)
ds2 = pygsti.io.read_dataset("../../../tutorial_files/Example_TinyDataset.txt")
```

```{code-cell} ipython3
#3) By generating fake data (using the smq1Q_XYI standard model module)
from pygsti.modelpacks import smq1Q_XYI

#Depolarize the perfect X,Y,I model
depol_gateset = smq1Q_XYI.target_model().depolarize(op_noise=0.1)

#Compute the sequences needed to perform Long Sequence GST on
# this Model with sequences up to length 128
exp_design = smq1Q_XYI.create_gst_experiment_design(max_max_length=128)
circuit_list = exp_design.all_circuits_needing_data

#Generate fake data
ds3 = pygsti.data.simulate_data(depol_gateset, circuit_list, num_samples=1000,
                                             sample_error='binomial', seed=100)
ds3b = pygsti.data.simulate_data(depol_gateset, circuit_list, num_samples=50,
                                              sample_error='binomial', seed=100)

#Package the ds3 and ds3b datasets together with their experiment design
# and save to disk for later tutorials to use for protocols
pygsti.protocols.ProtocolData(exp_design, ds3).write("../../../tutorial_files/Example_GST_Data")
pygsti.protocols.ProtocolData(exp_design, ds3b).write("../../../tutorial_files/Example_GST_Data_LowCnts")

#Also write the dataset files separately
pygsti.io.write_dataset("../../../tutorial_files/Example_Dataset.txt", ds3, outcome_label_order=['0','1'])
pygsti.io.write_dataset("../../../tutorial_files/Example_Dataset_LowCnts.txt", ds3b)
```

## Viewing data sets

Printing a `DataSet` is usually enough to see what's in it.

```{code-cell} ipython3
print("Dataset1:\n",ds1)
print("Dataset2:\n",ds2)
print("Dataset3 is too big to print, so here it is truncated to Dataset2's strings\n",
      ds3.truncate(list(ds2.keys())))
```

The `list(...)` there is not decoration. `keys()` is a generator, and under its default `missing_action='raise'` `truncate` consumes the argument once to look up indices and then reuses it as the key sequence. The second use sees an exhausted generator, so you get an empty `DataSet` back and no exception. Materialize the keys before passing them.

Notice that the outcome labels `'0'` and `'1'` print as `('0',)` and `('1',)`. Outcome labels in pyGSTi are tuples of time-ordered instrument element labels (see the [intermediate measurements tutorial](../gst/MidCircuitMeasurement)) followed by a POVM effect label. In the common case of no intermediate measurements, the label is a 1-tuple holding just the final effect label. You can write the bare effect label (`'0'`) almost anywhere such a label is expected and it gets converted to the 1-tuple internally; printing shows the 1-tuple to keep the general structure visible.

## Iterating over a data set

`keys()` returns a generator of `Circuit` objects.

```{code-cell} ipython3
ds1.keys()
```

There are many ways to walk the contents. Here's one.

```{code-cell} ipython3
for circuit in ds1.keys():
    dsRow = ds1[circuit]
    for outcome_label in dsRow.counts.keys():
        print("Circuit = %s, outcome label = %s, count = %d" % \
            (repr(circuit).ljust(13), str(outcome_label).ljust(7), dsRow[outcome_label]))
```

## Repeated circuits: the `collision_action` argument

What should happen when you add counts for a circuit that's already in the set? The `collision_action` argument to the `DataSet` constructor decides, and it takes one of three values.

- `"aggregate"` (the default) adds the new counts to the existing ones for matching outcomes.
- `"overwrite"` throws the old counts away and keeps only the new ones.
- `"keepseparate"` stores the second batch under a distinguishable key: the added circuit gets a nonzero `occurrence` ID (1 for the first repeat, 2 for the next, and so on), which shows up in printed output as a `@1` suffix.

```{code-cell} ipython3
ds_agg = pygsti.data.DataSet(outcome_labels=['0','1'], collision_action="aggregate") #the default
ds_agg.add_count_dict( ('Gx','Gy'), {'0': 10, '1': 90} )
ds_agg.add_count_dict( ('Gx','Gy'), {'0': 40, '1': 60} )
print("Aggregate-mode counts:", ds_agg[('Gx','Gy')].counts)

ds_ovr = pygsti.data.DataSet(outcome_labels=['0','1'], collision_action="overwrite")
ds_ovr.add_count_dict( ('Gx','Gy'), {'0': 10, '1': 90} )
ds_ovr.add_count_dict( ('Gx','Gy'), {'0': 40, '1': 60} )
print("Overwrite-mode counts:", ds_ovr[('Gx','Gy')].counts)

ds_sep = pygsti.data.DataSet(outcome_labels=['0','1'], collision_action="keepseparate")
ds_sep.add_count_dict( ('Gx','Gy'), {'0': 10, '1': 90} )
ds_sep.add_count_dict( ('Gx','Gy'), {'0': 40, '1': 60} )
print("Keepseparate-mode Dataset:\n",ds_sep)
```

Aggregation happens at the accessor, not in storage. Under the hood the second batch goes in with a later timestamp, so the aggregate set no longer has trivial time dependence and printing it gives the time-series view rather than the one-line summary the other two modes print. See for yourself.

```{code-cell} ipython3
print("Aggregate-mode Dataset:\n", ds_agg)
```

The occurrence ID is part of the key, so iterating over `keys()` gives you both rows and `circuit.occurrence` tells them apart.

```{code-cell} ipython3
for circuit in ds_sep.keys():
    print("occurrence =", circuit.occurrence, " counts =", ds_sep[circuit].counts)
```

To index a particular occurrence directly, build an editable `Circuit`, set its `occurrence`, and freeze it.

```{code-cell} ipython3
c_occ1 = pygsti.circuits.Circuit( ('Gx','Gy'), editable=True )
c_occ1.occurrence = 1
c_occ1.done_editing()
print(ds_sep[c_occ1].counts)
```

## Several passes of the same experiment: MultiDataSet

Sometimes you have several sets of counts over the *same* circuits: GST data taken on Monday and again on Tuesday, or a re-run after adjusting the apparatus. `pygsti.data.MultiDataSet` exists for this. It looks and acts like a dictionary of `DataSet` objects but stores a single shared list of circuits rather than one copy per contained set, which saves space and memory. It can also sum all its data into one `DataSet` via `datasets_aggregate(...)`, which is how you combine several passes of an experiment.

Three things to know:

- Add member sets with `add_dataset` (or plain bracket assignment), and only *static* `DataSet` objects are accepted. A `MultiDataSet` has to keep every member locked to the same circuits, and a non-static member could add or remove its own. If the set you want to add isn't static, call its `done_adding_data` first.
- Square-bracket indexing accesses the `MultiDataSet` as if it were a dictionary of `DataSet`s, and `len` follows that dictionary reading: it counts member sets, not circuits.
- A `MultiDataSet` reads and writes from a single text file with one column group per contained set; see `pygsti.io.read_multidataset`.

```{code-cell} ipython3
multiDS = pygsti.data.MultiDataSet()

#Create some datasets
ds_pass1 = pygsti.data.DataSet(outcome_labels=['0','1'])
ds_pass1.add_count_dict( (), {'0': 10, '1': 90} )
ds_pass1.add_count_dict( ('Gx',), {'0': 10, '1': 90} )
ds_pass1.add_count_dict( ('Gx','Gy'), {'0': 20, '1': 80} )
ds_pass1.add_count_dict( ('Gx','Gx','Gx','Gx'), {'0': 20, '1': 80} )
ds_pass1.done_adding_data()

ds_pass2 = pygsti.data.DataSet(outcome_labels=['0','1'])
ds_pass2.add_count_dict( (), {'0': 15, '1': 85} )
ds_pass2.add_count_dict( ('Gx',), {'0': 5, '1': 95} )
ds_pass2.add_count_dict( ('Gx','Gy'), {'0': 30, '1': 70} )
ds_pass2.add_count_dict( ('Gx','Gx','Gx','Gx'), {'0': 40, '1': 60} )
ds_pass2.done_adding_data()

multiDS['myDS'] = ds_pass1
multiDS['myDS2'] = ds_pass2

nMembers = len(multiDS)  # note: the number of member DataSets, not the number of circuits
dslabels = list(multiDS.keys())
nCircuits = len(list(multiDS['myDS'].keys()))
print("MultiDataSet has %d member DataSets over %d circuits, with labels %s"
      % (nMembers, nCircuits, dslabels))

for dslabel in multiDS:
    ds = multiDS[dslabel]
    print("Empty string data for %s = " % dslabel, ds[()])

for ds in multiDS.values():
    print("Gx string data (no label) =", ds[('Gx',)])

for dslabel,ds in multiDS.items():
    print("GxGy string data for %s =" % dslabel, ds[('Gx','Gy')])

dsSum = multiDS.datasets_aggregate('myDS','myDS2')
print("\nSummed data:")
print(dsSum)
```

The text format carries each member set's columns side by side, and a member may be specified by frequencies plus a count total rather than raw counts.

```{code-cell} ipython3
multi_dataset_txt = \
"""## Columns = DS0 0 count, DS0 1 count, DS1 0 frequency, DS1 count total
{} 0 100 0 100
Gx 10 90 0.1 100
GxGy 40 60 0.4 100
Gx^4 20 80 0.2 100
"""

with open("../../../tutorial_files/TinyMultiDataset.txt","w") as output:
    output.write(multi_dataset_txt)
multiDS_fromFile = pygsti.io.read_multidataset("../../../tutorial_files/TinyMultiDataset.txt", cache=False)

print("\nLoaded from file:\n")
print(multiDS_fromFile)
```

## Related topics

A `DataSet` can also store **time-dependent data**, with a timestamp attached to every count. That's covered in the [timestamped data tutorial](../drift/TimestampedData). Beyond what's shown here, the docstrings on `DataSet` and `MultiDataSet` methods are the reference.
