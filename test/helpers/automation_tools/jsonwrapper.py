import json

def read_json(filename):
    try:
        with open(filename, 'r') as jsonfile:
            return json.load(jsonfile)
    except:
        return {}

def write_json(data, filename):
    with open(filename, 'w') as jsonfile:
        json.dump(data, jsonfile)
