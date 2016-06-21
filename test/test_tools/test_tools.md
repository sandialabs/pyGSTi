# Automated testing tools

##### Generating package info:
Most package info can be generated with the command:  
`$ python genPackageInfo.py `,  
which takes excluded packages as arguments, and optionally the flags `--infoType` and `--changedOnly`  
By default, information about both package coverage and time taken is generated for every package:
Package | Coverage | Time
--- | --- | ---
objects| 77% | 83.24 s
drivers | 99% | 1936 s
However,
`python genPackageInfo.py report drivers` would generate info for every package except report and drivers  

The flag `infoType` can be set to either `coverage` or `benchmark`, so the command:  
`python genPackageInfo.py --infoType=coverage` would only generate coverage information, ex:
Package | Coverage
--- | ---
objects| 77%
drivers | 99%
(By default, this is sent to `objects/moduleinfo.out`)
a single    

When `changedOnly` is set to `True` (ex: `python genPackageInfo.py --changedOnly=True`), info for only the changed files will be generated

The scripts `runBenchmark.py` and `getCoverage.py` can also be run to generate output, ex:

`$ python getCoverage.py testAlgorithms.py construction --output=output/coverage.out --package=pygsti` would create a coverage file for both `testAlgorithms.py` and construction

`$ python runBenchmark.py testPrinter.py io --output=output/bench.out` would create a benchmark file for `testPrinter.py` and the io package
