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

# Time-dependent Data Sets
The [DataSet tutorial](DataSet) covered the basics of how to use `DataSet` objects with time-independent counts. When your data is time-stamped, either for each individual count or by groups of counts, there are additional (richer) options for analysis.  The `DataSet` class is also capable of storing time-dependent data by holding *series* of count data rather than binned numbers-of-counts, which are added via its `add_series_data` method.  Outcome counts are input by giving at least two parallel arrays of 1) outcome labels and 2) time stamps.  Optionally, one can provide a third array of repetitions, specifying how many times the corresponding outcome occurred at the time stamp.  While in reality no two outcomes are taken at exactly the same time, a `DataSet` allows for arbitrarily *coarse-grained* time-dependent data in which multiple outcomes are all tagged with the *same* time stamp.  In fact, the "time-independent" case considered in the aforementioned tutorial is actually just a special case in which the all data is stamped at *time=0*.

Below we demonstrate how to create and initialize a `DataSet` using time series data.

```{code-cell} ipython3
import pygsti

#Create an empty dataset                                                                       
tdds = pygsti.data.DataSet(outcome_labels=['0','1'])

#Add a "single-shot" series of outcomes, where each spam label (outcome) has a separate time stamp
tdds.add_raw_series_data( ('Gx',), #gate sequence                                                                 
            ['0','0','1','0','1','0','1','1','1','0'], #spam labels                                                                                                                 
            [0.0, 0.2, 0.5, 0.6, 0.7, 0.9, 1.1, 1.3, 1.35, 1.5]) #time stamps                                                                                              

#When adding outcome-counts in "chunks" where the counts of each
# chunk occur at nominally the same time, use 'add_raw_series_data' to
# add a list of count dictionaries with a timestamp given for each dict:
tdds.add_series_data( ('Gx','Gx'),  #gate sequence                                                               
                      [{'0':10, '1':90}, {'0':30, '1':70}], #count dicts                                                         
                      [0.0, 1.0]) #time stamps - one per dictionary                                                               

#For even more control, you can specify the timestamp of each count
# event or group of identical outcomes that occur at the same time:
#Add 3 'plus' outcomes at time 0.0, followed by 2 'minus' outcomes at time 1.0
tdds.add_raw_series_data( ('Gy',),  #gate sequence                                                               
                      ['0','1'], #spam labels                                                         
                      [0.0, 1.0], #time stamps                                                               
                      [3,2]) #repeats  

#The above coarse-grained addition is logically identical to:
# tdds.add_raw_series_data( ('Gy',),  #gate sequence                                                               
#                       ['0','0','0','1','1'], #spam labels                                                         
#                       [0.0, 0.0, 0.0, 1.0, 1.0]) #time stamps                                                               
# (However, the DataSet will store the coase-grained addition more efficiently.) 
```

When one is done populating the `DataSet` with data, one should still call `done_adding_data`:

```{code-cell} ipython3
tdds.done_adding_data()
```

Access to the underlying time series data is done by indexing on the gate sequence (to get a `DataSetRow` object, just as in the time-independent case) which has various methods for retrieving its underlying data:

```{code-cell} ipython3
tdds_row = tdds[('Gx',)]
print("INFO for Gx string:\n")
print( tdds_row )
      
print( "Raw outcome label indices:", tdds_row.oli )
print( "Raw time stamps:", tdds_row.time )
print( "Raw repetitions:", tdds_row.reps )
print( "Number of entries in raw arrays:", len(tdds_row) )

print( "Outcome Labels:", tdds_row.outcomes )
print( "Repetition-expanded outcome labels:", tdds_row.expanded_ol )
print( "Repetition-expanded outcome label indices:", tdds_row.expanded_oli )
print( "Repetition-expanded time stamps:", tdds_row.expanded_times )
print( "Time-independent-like counts per spam label:", tdds_row.counts )
print( "Time-independent-like total counts:", tdds_row.total )
print( "Time-independent-like spam label fraction:", tdds_row.fractions )

print("\n")

tdds_row = tdds[('Gy',)]
print("INFO for Gy string:\n")
print( tdds_row )
      
print( "Raw outcome label indices:", tdds_row.oli )
print( "Raw time stamps:", tdds_row.time )
print( "Raw repetitions:", tdds_row.reps )
print( "Number of entries in raw arrays:", len(tdds_row) )

print( "Spam Labels:", tdds_row.outcomes )
print( "Repetition-expanded outcome labels:", tdds_row.expanded_ol )
print( "Repetition-expanded outcome label indices:", tdds_row.expanded_oli )
print( "Repetition-expanded time stamps:", tdds_row.expanded_times )
print( "Time-independent-like counts per spam label:", tdds_row.counts )
print( "Time-independent-like total counts:", tdds_row.total )
print( "Time-independent-like spam label fraction:", tdds_row.fractions )
```

## Text data-file formats

It is possible to read text-formatted time-dependent data in two ways.

The first way is for the special case when
1. the outcomes are all single-shot 
2. the time stamps of the outcomes are the integers (starting at zero) for *all* of the operation sequences.
This corresponds to the case when each sequence is performed and measured simultaneously at equally spaced intervals. This is a bit fictitous, but it allows for the compact format given below.  Currently, the only way to read in this format is using the separate `read_time_dependent_dataset` function:

```{code-cell} ipython3
tddataset_txt = \
"""## 0 = 0                                                                                                                   
## 1 = 1                                                                                                                      
{} 011001                                                                                                                     
Gx 111000111                                                                                                                  
Gy 11001100                                                                                                                   
"""
with open("../../tutorial_files/TDDataset.txt","w") as output:
    output.write(tddataset_txt)
tdds_fromfile = pygsti.io.read_time_dependent_dataset("../../tutorial_files/TDDataset.txt")
print(tdds_fromfile)

print("Some tests:")
print(tdds_fromfile[()].fractions['1'])
print(tdds_fromfile[('Gy',)].fractions['1'])
print(tdds_fromfile[('Gx',)].total)
```

The second way can describe arbitrary timstamped data, and uses a more general format where each circuit is on a line by itself, followed by two or three subsequent lines giving the timestamps, the outcome labels, and (optionally) the repetition counts for that circuit.  If the repetition counts are not given, they are all assumed to equal 1.  This is the format that is needed to interact with nicely with `ProtocolData` objects, e.g. for use with `load_data_from_dir`.  Here's an example that creates the same `DataSet` as the one loaded in above, and then loads it in using the usual `load_dataset` function:

```{code-cell} ipython3
general_tddataset_txt = \
"""{}
times: 0  1  2  3  4  5
outcomes: 0  1  1  0  0  1

Gx
times: 0  1  2  3  4  5  6  7  8
outcomes: 1  1  1  0  0  0  1  1  1

Gy
times: 0  1  2  3  4  5  6  7
outcomes: 1  1  0  0  1  1  0  0

"""
with open("../../tutorial_files/DatasetWithTimestamps.txt","w") as output:
    output.write(general_tddataset_txt)

#This format can be read in using the usual 'load'
general_tdds_fromfile = pygsti.io.read_dataset("../../tutorial_files/DatasetWithTimestamps.txt")
print(general_tdds_fromfile)
```

The `DatasetWithTimestamps.txt` file could also have been created by specifying `fixed_column_mode=False` to the usual `write_dataset` function, that is:

```{code-cell} ipython3
pygsti.io.write_dataset("../../tutorial_files/DatasetWithTimestamps.txt", tdds_fromfile, fixed_column_mode=False)
```

If you're recording several passes through a set of circuits, and all the data on each pass is considered to occur at the same time (i.e. a course-graining of the time-stamped data), then it may be useful to specify the repetition counts.  For example, the following data file describes data that was taken in two passes (at time 1.0 and 2.0) of 100 circuit repetitions:

```{code-cell} ipython3
general_tddataset_txt = \
"""{}
times:        1  1  2  2
outcomes:     0  1  0  1
repetitions: 20 80 25 75

Gx
times:        1  1  2  2
outcomes:     0  1  0  1
repetitions: 50 50 55 45

Gy
times:        1  1  2  2
outcomes:     0  1  0  1
repetitions: 63 37 52 48

"""
with open("../../tutorial_files/DatasetWith2Passes.txt","w") as output:
    output.write(general_tddataset_txt)

#This format can be read in using the usual 'load'
twopass_ds = pygsti.io.read_dataset("../../tutorial_files/DatasetWith2Passes.txt")
print(twopass_ds)

print("Some tests:")
print(twopass_ds[()].counts)  #total counts (aggregates over time)
print(twopass_ds[()][1.0])    # counts at time=0.0 -- Note this must be a *float* or it's interpeted as in index
print(twopass_ds[()][2.0])    # counts at time=1.0

#fraction and total function act like they would if all the data was aggregated.
print(twopass_ds[()].fractions['1'])
print(twopass_ds[('Gy',)].fractions['1'])
print(twopass_ds[('Gx',)].total)
```


