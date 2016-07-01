# Automated testing tools

##### Generating package info:
Most package info can be generated with the command:  
`$ python genPackageInfo.py `,  
which takes excluded packages as arguments, and optionally the flags `--infoType` and `--changedOnly`  
By default, information about both package coverage and time taken is generated for every package:

| Package       | Coverage      | Time  |
| ------------- |:-------------:| -----:|
| objects       | 77%           | 83.24s |
| drivers       | 99%           | 1936s |

However,
`python genPackageInfo.py report drivers` would generate info for every package except report and drivers  

The flag `infoType` can be set to either `coverage` or `benchmark`, so the command:  
`python genPackageInfo.py --infoType=coverage` would only generate coverage information, ex:

| Package       | Coverage      |
| ------------- |:-------------:|
| objects       | 77%           |
| drivers       | 99%           |

(By default, this is sent to `objects/packageinfo.out`)

When `changedOnly` is set to `True` (ex: `python genPackageInfo.py --changedOnly=True`), info for only the changed files will be generated

The scripts `runBenchmark.py` and `getCoverage.py` can also be run to generate output, ex:

`$ python getCoverage.py testAlgorithms.py construction --output=output/coverage.out --package=pygsti` would create a coverage file for both `testAlgorithms.py` and construction

`$ python runBenchmark.py testPrinter.py io --output=output/bench.out` would create a benchmark file for `testPrinter.py` and the io package


##### Automated testing speedups/improvements

Two scripts, `runChanged.py` and `runPackage.py` offer tools to speed up testing:  
Both scripts accept the flag `--lastFailed=True`, which, if a test has recently run, will only run the tests that previously failed.  
ex `python runPackage.py io --lastFailed=True` would only run the tests that previously failed in the io test package

Also, `runChanged.py` will only run tests covering packages that have changed since the latest commit (so, `python runChanged.py --lastFailed=True` would only run the tests that have failed in the packages that have been changed since the last commit)

The lastFailed functionality stores a pickle file in each test package (``'last_failed.pkl'``), which outlines which tests have failed in the latest run

lastFailed wont track any errors that belong to the unittest itself. For example, if a unittest relied on the `sys` module, but forgot to import it, the error would be unconnected to any of the test cases, and impossible to track.
