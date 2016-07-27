#!/usr/bin/env python3
from helpers.pylint           import get_score, look_for, find_warnings, find_errors, run_adjustables, lint_all
from helpers.automation_tools import read_yaml, write_yaml, get_args
import sys
import argparse


def check_score(on=lint_all):
    yamlFile = 'config/pylint_config.yml'

    config       = read_yaml(yamlFile)
    desiredScore = config['desired-score']
    print('Score should be: %s' % desiredScore)
    score        = get_score(on)
    print('Score was: %s' % score)

    if on == lint_all:
        if float(score) >= float(desiredScore):
            config['desired-score'] = score # Update the score if it is higher than the last one
            write_yaml(config, yamlFile)
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
    parser.add_argument('--score', '-s', type=bool,
                        help='check the current repo\'s score against the last-highest score')
    parser.add_argument('--errors', '-e', type=bool,
                        help='check for errors in the repo')
    parser.add_argument('--noerrors', '-n', type=bool,
                        help='fail if any errors are found in core pygsti repo (not tests)')
    parser.add_argument('--warnings', '-w', type=bool,
                        help='check for warnings in the repo')
    parser.add_argument('--adjustables', '-a', type=bool,
                        help='check for refactors in the repo')

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
        look_for(parsed.specific)
