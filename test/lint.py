#!/usr/bin/env python
from helpers.pylint           import get_score, look_for, find_warnings, find_errors, run_adjustables
from helpers.automation_tools import read_yaml, write_yaml, get_args
import sys

def check_score():
    yamlFile = 'config/pylint_config.yml'

    config       = read_yaml(yamlFile)
    desiredScore = config['desired-score']
    print('Score should be: %s' % desiredScore)
    score        = get_score()
    print('Score was: %s' % score)

    if float(score) >= float(desiredScore):
        config['desired-score'] = score # Update the score if it is higher than the last one
        write_yaml(config, yamlFile)
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    args, kwargs = get_args(sys.argv)
    # No arguments specified
    if 'score' in kwargs:
        check_score()
    if 'errors' in kwargs:
        find_errors()
    if 'warnings' in kwargs:
        find_warnings()
    if 'adjustables' in kwargs:
        run_adjustables()
    if len(args) > 0:
        look_for(args)
