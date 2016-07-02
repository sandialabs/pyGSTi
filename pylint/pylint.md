# Pylint

### Standard scripts

##### findErrors.py

Used to find errors in the entire pygsti repository. Individual errors are blacklisted in `config.yml`

##### findWarnings.py

Used to find warnings. Blacklisting is also in `config.yml`

##### runAdjustables.py  

Takes the list of adjustable refactors in `config.yml`, and generates a file for each.
(Guarantees the filelength is < 20 lines, adjusting parameters where needed)

##### lintAll.py  

Find Errors, Warnings, and whitelisted refactors and conventions throughout the repository

### Specialized scripts

#### Specialized scripts use `lookFor.py`, which allows linting of a specific item

##### Usage:

`./lookFor.py filename item` -> `./lookFor.py missingdocs missing-docstring`

A full list of items can be found [here](https://docs.pylint.org/features.html#general-options)

(Or in `pylint_message.txt`)

#### Specializations:

##### genDuplicateCode.sh  
 (`./lookFor.py duplicate-code duplicate-code`)

##### genMissingDocs.sh
(`./lookFor.py missingdocs missing-docstring empty-docstring`)

##### genTODOS.sh
(`./lookFor.py todos fixme`)

##### genUnused.sh
 (`./lookFor.py unused unused-variable unused-import unused-argument`)
