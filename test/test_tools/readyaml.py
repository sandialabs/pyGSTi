from __future__ import print_function
import yaml

def read_yaml(filename):
    with open(filename, 'r') as yamlfile:
        try:
            return yaml.load(yamlfile)
        except yaml.YAMLError as exc:
            print(exc)
