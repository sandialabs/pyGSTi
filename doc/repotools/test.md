# Testing tools

### By default, `./runTests.py` runs tests for all packages.


##### Optional flags:

| Flag            | Description                                          |
|-----------------|:-----------------------------------------------------|
| `--version=2.7` | specifies python version (which should be 3 by now!) |
| `--changed`     | run packages that have changed since last git commit |
| `--fast`        | doesn't run `report` or `drivers`                    |
| `--failed`      | runs tests that failed the previous time             |
| `--parallel`    | runs tests across multiple cores                     |
| `--cores=4`     | runs tests across n cores (max if left out)          |

##### Also, `runTests.py` can take individual packages/files/tests as arguments
ex:  *`./runTests.py tools io`* Runs only the `tools` and `io` packages
*`./runTests.py report/testReport.py`* runs only tests in `report`
*`./runTests.py report/testReport.py:TestReport.test_reports_logL_TP_wCIs`* runs the specific test `test_reports_logL_TP_wCIs`, which is a method of the test case `TestReport`

`runTests.py` now uses nose by default
(So, the above example of `./runTests.py report/testReport.py` would expand to:
`python3.5 -m nose report/testReport.py`)
  
