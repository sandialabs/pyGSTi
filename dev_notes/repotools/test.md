# Testing tools

### By default, `./runTests.py` runs tests for all packages.

*Warning*. This file was designed around nosetests. When we converted to pytest
we didn't port all the available functionality (particularly the "parallel"
option). Correct behavior of coverage reporting has not been verified.

##### Optional flags:

| Flag            | Description                                          |
|-----------------|:-----------------------------------------------------|
| `--version=2.7` | specifies python version (which should be 3 by now!) |
| `--changed`     | run packages that have changed since last git commit |
| `--fast`        | doesn't run `report` or `drivers`                    |
| `--failed`      | runs tests that failed the previous time             |
| `--parallel`    | runs tests across multiple cores                     |
| `--cores=4`     | runs tests across n cores (max if left out)          |
| `--nocover`     | skip coverage tests                                  |
| `--output`      | filename to direct output to                         |

The `runTests.py` script also takes the flag `--help` , which will show all possible arguments

##### Also, `runTests.py` can take individual packages/files/tests as arguments
ex:  *`./runTests.py tools io`* Runs only the `tools` and `io` packages   
*`./runTests.py report/testReport.py`* runs only tests in `report`  
*`./runTests.py report/testReport.py:TestReport.test_reports_logL_TP_wCIs`* runs the specific test `test_reports_logL_TP_wCIs`, which is a method of the test case `TestReport`

`runTests.py` now uses pytest by default  
(So, the above example of `./runTests.py report/testReport.py` would expand to:  
`pytest report/testReport.py`)  


 - Current test coverage is outlined in test/coverage_status.txt
 - Test coverage status will be generated in travis CI logs under the [beta branch](https://travis-ci.org/pyGSTio/pyGSTi/branches)
 - Coverage is generated automatically when `./runTests.py` is used

  
