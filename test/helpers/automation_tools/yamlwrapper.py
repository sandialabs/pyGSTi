from __future__ import print_function
import ruamel.yaml as yaml

def read_yaml(filename):
    with open(filename, 'r') as yamlfile:
        try:
            dictionary = yaml.load(yamlfile, yaml.RoundTripLoader)
            if dictionary is None: # if file is empty
                dictionary = {}
            return dictionary
        except yaml.YAMLError as exc:
            print(exc)

def write_yaml(yamldata, filename):
    with open(filename, 'w') as outfile:
        outfile.write( yaml.dump(yamldata, default_flow_style=False, Dumper=yaml.RoundTripDumper) )
