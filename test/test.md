# Testing tools

### By default, ./runTest.py runs tests for all packages.

##### Optional flags:

| Flag            | Description                                          |
|-----------------|:-----------------------------------------------------|
| `--nose`        | appends ` -m nose` to the python command             |
| `--version=2.7` | specifies python version                             |
| `--changed`     | run packages that have changed since last git commit |
| `--fast-only`   | doesn't run `report` or `drivers`                    |
| `--lastFailed`  | runs tests that failed the previous time             |

##### Also, `runTests.py` can take individual packages as arguments
ex:  *`./runTests.py tools io`* Runs only the `tools` and `io` packages

### Generation package info:

##### If `runTests.py` is given the flag `--info`, it will create tables that are saved in `output/packageinfo.out`

ex: Information about both package coverage and time taken is generated for every package:

| Package       | Coverage      | Time   |
| ------------- |:--------------|:-------|
| objects       | 77%           | 83.24s |
| drivers       | 99%           | 1936s  |
| ...           | ...           | ...    |

##### This command takes the same flags as above, but `--nose`, `--version`, and `--lastFailed` have no effect.
