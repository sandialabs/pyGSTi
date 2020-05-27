#!/usr/bin/env python3
import pickle
import pygsti
import uuid

from pprint import pprint

from pygsti.tools import timed_block
import pygsti.construction as pc


def main():
    gates = ['Gi','Gx','Gy']
    fiducials = pc.gatestring_list([ (), ('Gx',), ('Gy',), ('Gx','Gx'), ('Gx','Gx','Gx'), ('Gy','Gy','Gy') ]) # fiducials for 1Q MUB
    germs = pc.gatestring_list( [('Gx',), ('Gy',), ('Gi',), ('Gx', 'Gy',),
                                 ('Gx', 'Gy', 'Gi',), ('Gx', 'Gi', 'Gy',),('Gx', 'Gi', 'Gi',), ('Gy', 'Gi', 'Gi',),
                                 ('Gx', 'Gx', 'Gi', 'Gy',), ('Gx', 'Gy', 'Gy', 'Gi',),
                                 ('Gx', 'Gx', 'Gy', 'Gx', 'Gy', 'Gy',)] )
    maxLengths = [1,2,4,8,16,32,64,128,256]    
    lsgst_lists = pc.create_lsgst_circuits(gates, fiducials, fiducials, germs, maxLengths) 
    lsgst_tuple = tuple(lsgst_lists)
    iterations = 1000
    timeDict = dict()
    with timed_block('hash_gatestring_list', timeDict):
        for i in range(iterations):
            hash(lsgst_tuple)

    exampleUUID = uuid.uuid4()
    alt_hash = pygsti.tools.smartcache.digest
    with timed_block('digest_uuid', timeDict):
        for i in range(iterations):
            alt_hash(exampleUUID)

    print('Hashing gslist of length {} takes {} seconds on average'.format(
        len(lsgst_tuple),
        timeDict['hash_gatestring_list'] / iterations))
    print('UUID digest takes {} seconds on average'.format(
        timeDict['digest_uuid'] / iterations))

if __name__ == '__main__':
    main()
