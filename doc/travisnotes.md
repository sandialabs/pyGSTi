# Travis CI Notes - Lucas

### Overview:

  - Travis CI clones pygsti, and executes the `test/travisTests.py` script in different environments
  - A build matrix is created, testing almost everything it can in the least amount of time
  - If the tests pass on develop, an automatic push to the beta branch is made. This seems to be buggy.
  - On beta, coverage info is generated and should be visible in the job log
  - When a build fails, the author of the latest commit to that branch is notified by github-registered e-mail. There is also custom configuration in `.travis.yml`


### The build matrix:

Currently, our tests run both python 2.7 and 3.5:

  - Default: test `objects`, `tools`, `io`, `construction`, `optimize`, and `algorithms` packages.
  - ReportA: run `test_reports_logL_TP_wCIs` only (77% coverage of `report` package in ~7 minutes)
  - ReportB: run all report tests except for `testReport.py`. This includes formatter, table, analysis, metrics, and plotting tests.
  - Drivers: run only the `drivers` test package

This results in a 2x4 matrix, which often finishes completely in about thirty-forty minutes.

So, this means that in the test `testReport.py`, only `test_reports_logL_TP_wCIs` is run, and *none of the other* tests in `testReport.py`.

### Optimization

  - There is a script `CI/install.sh`, which only downloads packages required for the current test environment
    (For example, `texlive-full` is only installed for `ReportA` and `drivers` environments, since they need pdflatex)

  - Additionally, all pip installs are cached, generally making them run in under a few seconds. However, the cache size is enormous, at around 2Gb
    (The most recent beta build installed all pip dependencies in under 10 seconds - caching is worth keeping :) )

  - By default, all tests besides mpi tests run in parallel through the `nose` `--processes=-1` (all cores) flag.

  - Coverage is only turned on when testing on `beta`

### Automatic pushing:

Now, this uses a javascript travis-after-all, which more reliably pushes to beta. The only downside is that if a collective build goes over 50 minutes, it times out, even if the individual jobs were short.

