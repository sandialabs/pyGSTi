#!/usr/bin/env python3
import argparse
import sys

from helpers.automation_tools import read_json, write_json
from helpers.pylint import get_score, look_for, find_warnings, find_errors, run_adjustables, lint_all


# See pyGSTi/doc/pylint.md!

# Fail if the score is below a threshold
def check_score(on=lint_all):
    jsonFile = 'config/pylint_config.json'

    config       = read_json(jsonFile)
    desiredScore = config['desired-score']
    print('Score should be: %s' % desiredScore)
    score        = get_score(on)
    print('Score was: %s' % score)

    if on == lint_all:
        if float(score) >= float(desiredScore):
            config['desired-score'] = score # Update the score if it is higher than the last one
            write_json(config, jsonFile)
            sys.exit(0)
        else:
            sys.exit(1)
    else:
        if score < 10:
            sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Lint pygsti')
    parser.add_argument('specific', nargs='*',
                        help='lint for specific items')
    parser.add_argument('--score', '-s', action='store_true',
                        help='check the current repo\'s score against the last-highest score')
    parser.add_argument('--errors', '-e', action='store_true',
                        help='check for errors in the repo')
    parser.add_argument('--noerrors', '-n', action='store_true',
                        help='fail if any errors are found in core pygsti repo (not tests)')
    parser.add_argument('--warnings', '-w', action='store_true',
                        help='check for warnings in the repo')
    parser.add_argument('--adjustables', '-a', action='store_true',
                        help='check for refactors in the repo')
    parser.add_argument('--andtests', '-t', action='store_true',
                        help='include tests in specific items being linted for')


    parsed = parser.parse_args(sys.argv[1:])

    if parsed.score:
        check_score()
    if parsed.noerrors:
        check_score(find_errors) # Call find_errors to score by instead of lint_all
    elif parsed.errors:          # Doesn't need to run if no-errors already did
        find_errors(coreonly=False)
    if parsed.warnings:
        find_warnings()
    if parsed.adjustables:
        run_adjustables()
    if parsed.specific != []:
        look_for(parsed.specific, not parsed.andtests)
