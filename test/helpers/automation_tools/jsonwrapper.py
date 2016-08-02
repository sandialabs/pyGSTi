from __future__ import print_function
import json

def read_json(filename):
    with open(filename, 'r') as jsonfile:
        return json.load(jsonfile)

def write_json(data, filename):
    with open(filename, 'w') as jsonfile:
        json.dump(data, jsonfile)
