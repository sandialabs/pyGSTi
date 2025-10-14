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

# MultiDataSets

Sometimes it is useful to deal with several sets of data all of which hold counts for the *same* set of operation sequences.  For example, colleting data to perform GST on Monday and then again on Tuesday, or making an adjustment to an experimental system and re-taking data, could create two separate data sets with the same sequences.  PyGSTi has a separate data type, `pygsti.objects.MultiDataSet`, for this purpose.  A `MultiDataSet` looks and acts like a simple dictionary of `DataSet` objects, but underneath implements some certain optimizations that reduce the amount of space and memory required to store the data.  Primarily, it holds just a *single* list of the circuits - as opposed to an actual dictionary of `DataSet`s in which each `DataSet` contains it's own copy of the circuits.  In addition to being more space efficient, a `MultiDataSet` is able to aggregate all of its data into a single "summed" `DataSet` via `get_datasets_aggregate(...)`, which can be useful for combining several "passes" of experimental data.  

Several remarks regarding a `MultiDataSet` are worth mentioning:
- you add `DataSets` to a `MultiDataSet` using the `add_dataset` method.  However only *static* `DataSet` objects can be added.  This is because the MultiDataSet must keep all of its `DataSet`s locked to the same set of sequences, and a non-static `DataSet` allows the addition or removal of only *its* sequences.  (If the `DataSet` you want to add isn't in static-mode, call its `done_adding_data` method.)
- square-bracket indexing accesses the `MultiDataSet` as if it were a dictionary of `DataSets`.
- `MultiDataSets` can be loaded and saved from a single text-format file with columns for each contained `DataSet` - see `pygsti.io.load_multidataset`.

Here's a brief example of using a `MultiDataSet`:

```{code-cell} ipython3
import pygsti

multiDS = pygsti.data.MultiDataSet()

#Create some datasets                                           
ds = pygsti.data.DataSet(outcome_labels=['0','1'])
ds.add_count_dict( (), {'0': 10, '1': 90} )
ds.add_count_dict( ('Gx',), {'0': 10, '1': 90} )
ds.add_count_dict( ('Gx','Gy'), {'0': 20, '1': 80} )
ds.add_count_dict( ('Gx','Gx','Gx','Gx'), {'0': 20, '1': 80} )
ds.done_adding_data()

ds2 = pygsti.data.DataSet(outcome_labels=['0','1'])            
ds2.add_count_dict( (), {'0': 15, '1': 85} )
ds2.add_count_dict( ('Gx',), {'0': 5, '1': 95} )
ds2.add_count_dict( ('Gx','Gy'), {'0': 30, '1': 70} )
ds2.add_count_dict( ('Gx','Gx','Gx','Gx'), {'0': 40, '1': 60} )
ds2.done_adding_data()

multiDS['myDS'] = ds
multiDS['myDS2'] = ds2

nStrs = len(multiDS)
dslabels = list(multiDS.keys())
print("MultiDataSet has %d operation sequences and DataSet labels %s" % (nStrs, dslabels))
    
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

```{code-cell} ipython3
multi_dataset_txt = \
"""## Columns = DS0 0 count, DS0 1 count, DS1 0 frequency, DS1 count total                                
{} 0 100 0 100                                                                                                      
Gx 10 90 0.1 100                                                                                                    
GxGy 40 60 0.4 100                                                                                                  
Gx^4 20 80 0.2 100                                                                                                  
"""

with open("../../tutorial_files/TinyMultiDataset.txt","w") as output:
    output.write(multi_dataset_txt)
multiDS_fromFile = pygsti.io.read_multidataset("../../tutorial_files/TinyMultiDataset.txt", cache=False)

print("\nLoaded from file:\n")
print(multiDS_fromFile)
```

Those are the basics of using `MultiDataSet`.  More information is available in the docstrings for the various `MultiDataSet` methods.
