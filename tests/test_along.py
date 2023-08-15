#!/usr/bin/env python3

import sys
import numpy as np
sys.path.insert(1, '..')
import rsr


data = np.load('incircles_input_trim1.npz')
print("points: ", len(data['surf_amp']), "circles: ", len(data['query_lon']))
max_points = 3_000
#len(data['surf_amp']) - 10 #100_000
#max_circles = 1_000 #len(data['query_lon']) - 10 # 1000
data2 = {'radius': data['radius']}
#sys.exit(0)
assert len(data['surf_amp']) > max_points
for k in 'surf_amp SUB_SC_EAST_LONGITUDE SUB_SC_PLANETOCENTRIC_LATITUDE'.split():
    data2[k] = data[k][0:max_points]

#assert len(data['query_lon']) > max_circles
#for k in 'query_lon query_lat'.split():
#    data2[k] = data[k][0:max_points]

nbcores = -1
out = rsr.run.along(data2['surf_amp'], verbose=True, nbcores=nbcores)

with open('out_single.txt', 'wt') as fout:
    print(out, file=fout)

nbcores = 2
# def along(amp, nbcores=1, verbose=True, **kwargs):
out = rsr.run.along(data2['surf_amp'][0:max_points], verbose=True, nbcores=nbcores)

with open('out_multi.txt', 'wt') as fout:
    print(out, file=fout)

