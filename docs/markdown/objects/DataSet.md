---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Data Sets
This tutorial demonstrates how to create and use `DataSet` objects.  PyGSTi uses `DataSet` objects to hold experimental or simulated data in the form of outcome counts.  When a `DataSet` is used to hold time-independent data (the typical case, and all we'll look at in this tutorial), it essentially looks like a nested dictionary which associates operation sequences with dictionaries of (outcome-label,count) pairs so that `dataset[circuit][outcomeLabel]` can be used to read & write the number of `outcomeLabel` outcomes of the experiment given by the circuit `circuit`.

There are a few important differences between a `DataSet` and a dictionary-of-dictionaries:
- `DataSet` objects can be in one of two modes: *static* or *non-static*.  When in *non-static* mode, data can be freely modified within the set, making this mode to use during the data-entry.  In the *static* mode, data cannot be modified and the `DataSet` is essentially read-only.  The `done_adding_data` method of a `DataSet` switches from non-static to static mode, and should be called, as the name implies, once all desired data has been added (or modified).  Once a `DataSet` is static, it is read-only for the rest of its life; to modify its data the best one can do is make a non-static *copy* via the `copy_nonstatic` member and modify the copy.

- Because `DataSet`s may contain time-dependent data, the dictionary-access syntax for a single outcome label (i.e. `dataset[circuit][outcomeLabel]`) *cannot* be used to write counts for new `circuit` keys; One should instead  use the `add_`*xxx* methods of the `DataSet` object.

Once a `DataSet` is constructed, filled with data, and made *static*, it is typically passed as a parameter to one of pyGSTi's algorithm or driver routines to find a `Model` estimate based on the data.  This tutorial focuses on how to construct a `DataSet` and modify its data.  Later tutorials will demonstrate the different GST algorithms.

```{code-cell} ipython3
import pygsti
```

## Creating a `DataSet`
There three basic ways to create `DataSet` objects in `pygsti`:
* By creating an empty `DataSet` object and manually adding counts corresponding to operation sequences.  Remember that the `add_`*xxx* methods must be used to add data for operation sequences not yet in the `DataSet`.  Once the data is added, be sure to call `done_adding_data`, as this restructures the internal storage of the `DataSet` to optimize the access operations used by algorithms.
* By loading from a text-format dataset file via `pygsti.io.read_dataset`.  The result is a ready-to-use-in-algorithms *static* `DataSet`, so there's no need to call `done_adding_data` this time.
* By using a `Model` to generate "fake" data via `generate_fake_data`. This can be useful for doing simulations of GST, and comparing to your experimental results.

We do each of these in turn in the cells below.

```{code-cell} ipython3
#1) Creating a data set from scratch
#    Note that tuples may be used in lieu of OpString objects
ds1 = pygsti.data.DataSet(outcome_labels=['0','1'])
ds1.add_count_dict( ('Gx',), {'0': 10, '1': 90} )
ds1.add_count_dict( ('Gx','Gy'), {'0': 40, '1': 60} )
ds1[('Gy',)] = {'0': 10, '1': 90} # dictionary assignment

#Modify existing data using dictionary-like access
ds1[('Gx',)]['0'] = 15
ds1[('Gx',)]['1'] = 85

#OpString objects can be used.
mdl = pygsti.circuits.Circuit( ('Gx','Gy'))
ds1[mdl]['0'] = 45
ds1[mdl]['1'] = 55

ds1.done_adding_data()
```

```{code-cell} ipython3
#2) By creating and loading a text-format dataset file.  The first
#    row is a directive which specifies what the columns (after the
#    first one) holds.  Note that "0" and "1" in are the 
#    SPAM labels and must match those of any Model used in 
#    conjuction with this DataSet.
dataset_txt = \
"""## Columns = 0 count, 1 count
{}@(0)             0 100
Gxpi2:0@(0)        10 90
Gxpi2:0Gypi2:0@(0) 40 60
Gxpi2:0^4@(0)      20 90
"""
with open("../tutorial_files/Example_TinyDataset.txt","w") as tinydataset:
    tinydataset.write(dataset_txt)
ds2 = pygsti.io.read_dataset("../tutorial_files/Example_TinyDataset.txt")
```

```{code-cell} ipython3
#3) By generating fake data (using the std1Q_XYI standard model module)
from pygsti.modelpacks import smq1Q_XYI

#Depolarize the perfect X,Y,I model
depol_gateset = smq1Q_XYI.target_model().depolarize(op_noise=0.1)

#Compute the sequences needed to perform Long Sequence GST on 
# this Model with sequences up to lenth 128
exp_design = smq1Q_XYI.create_gst_experiment_design(max_max_length=128)
circuit_list = exp_design.all_circuits_needing_data

#Generate fake data (Tutorial 00)
ds3 = pygsti.data.simulate_data(depol_gateset, circuit_list, num_samples=1000,
                                             sample_error='binomial', seed=100)
ds3b = pygsti.data.simulate_data(depol_gateset, circuit_list, num_samples=50,
                                              sample_error='binomial', seed=100)

#Package the ds3 and ds3b datasets together with their experiment design
# and save to disk for later tutorials to use for protocols
pygsti.protocols.ProtocolData(exp_design, ds3).write("../tutorial_files/Example_GST_Data")
pygsti.protocols.ProtocolData(exp_design, ds3b).write("../tutorial_files/Example_GST_Data_LowCnts")

#Also write the dataset files separately
pygsti.io.write_dataset("../tutorial_files/Example_Dataset.txt", ds3, outcome_label_order=['0','1']) 
pygsti.io.write_dataset("../tutorial_files/Example_Dataset_LowCnts.txt", ds3b) 
```

## Viewing `DataSets`

```{code-cell} ipython3
#It's easy to just print them:
print("Dataset1:\n",ds1)
print("Dataset2:\n",ds2)
print("Dataset3 is too big to print, so here it is truncated to Dataset2's strings\n", ds3.truncate(ds2.keys()))
```

**Note that the outcome labels `'0'` and `'1'` appear as `('0',)` and `('1',)`**.  This is because outcome labels in pyGSTi are tuples of time-ordered instrument element (see the [intermediate measurements tutorial](advanced/Instruments.ipynb)) and POVM effect labels.  In the special but common case when there are no intermediate measurements, the outcome label is a 1-tuple of just the final POVM effect label.  In this case, one may use the effect label itself (e.g. `'0'` or `'1'`) in place of the 1-tuple in almost all contexts, as it is automatically converted to the 1-tuple (e.g. `('0',)` or `('1',)`) internally.  When printing, however, the 1-tuple is still displayed to remind the user of the more general structure contained in the `DataSet`.

+++

## Iteration over data sets

```{code-cell} ipython3
# A DataSet's keys() method returns a generator of OpString objects
ds1.keys()
```

```{code-cell} ipython3
# There are many ways to iterate over a DataSet.  Here's one:
for circuit in ds1.keys():
    dsRow = ds1[circuit]
    for spamlabel in dsRow.counts.keys():
        print("Circuit = %s, SPAM label = %s, count = %d" % \
            (repr(circuit).ljust(13), str(spamlabel).ljust(7), dsRow[spamlabel]))
```

## Advanced features of data sets

### `collision_action` argument
When creating a `DataSet` one may specify the `collision_action` argument as either `"aggregate"` (the default) or `"keepseparate"`.  The former instructs the `DataSet` to simply add the counts of like outcomes when counts are added for an already existing gate sequence.  `"keepseparate"`, on the other hand, causes the `DataSet` to tag added count data by appending a fictitious `"#<n>"` operation label to a gate sequence that already exists, where `<n>` is an integer.  When retreiving the keys of a `keepseparate` data set, the `stripOccuranceTags` argument to `keys()` determines whether the `"#<n>"` labels are included in the output (if they're not - the default - duplicate keys may be returned).  Access to different occurances of the same data are provided via the `occurrance` argument of the `get_row` and `set_row` functions, which should be used instead of the usual bracket indexing.

```{code-cell} ipython3
ds_agg = pygsti.data.DataSet(outcome_labels=['0','1'], collision_action="aggregate") #the default
ds_agg.add_count_dict( ('Gx','Gy'), {'0': 10, '1': 90} )
ds_agg.add_count_dict( ('Gx','Gy'), {'0': 40, '1': 60} )
print("Aggregate-mode Dataset:\n",ds_agg)

ds_sep = pygsti.data.DataSet(outcome_labels=['0','1'], collision_action="keepseparate")
ds_sep.add_count_dict( ('Gx','Gy'), {'0': 10, '1': 90} )
ds_sep.add_count_dict( ('Gx','Gy'), {'0': 40, '1': 60} )
print("Keepseparate-mode Dataset:\n",ds_sep)
```

## Related topics
This concludes our overview of `DataSet` objects.  Here are a couple of related topics we didn't touch on that might be of interest:
- the ability of a `DataSet` to store **time-dependent data** (in addition to the count data described above) is covered in the [timestamped data tutorial](advanced/TimestampedDataSets.ipynb).
- the `MultiDataSet` object, which is similar to a dictionary of `DataSets` is described in the [MultiDataSet tutorial](advanced/MultiDataSet.ipynb).  `MultiDataSet` objects are useful when you have several data sets containing outcomes for the *same* set of circuits, like when you do multiple passes of the same experimental procedure.

```{code-cell} ipython3

```
