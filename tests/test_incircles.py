#!/usr/bin/env python3

import sys
import numpy as np
sys.path.insert(1, '..')
import rsr


data = np.load('incircles_input_trim1.npz')
print("points: ", len(data['surf_amp']), "circles: ", len(data['query_lon']))
nbcores = -1
out = rsr.run.incircles(data['surf_amp'], data['SUB_SC_EAST_LONGITUDE'],
                        data['SUB_SC_PLANETOCENTRIC_LATITUDE'],
                        data['query_lon'], data['query_lat'], data['radius'], verbose=True, nbcores=nbcores)


with open('out_single.txt', 'wt') as fout:
    print(out, file=fout)

nbcores = 2
out = rsr.run.incircles(data['surf_amp'], data['SUB_SC_EAST_LONGITUDE'],
                        data['SUB_SC_PLANETOCENTRIC_LATITUDE'],
                        data['query_lon'], data['query_lat'], data['radius'], verbose=True, nbcores=nbcores)

with open('out_multi.txt', 'wt') as fout:
    print(out, file=fout)

