#!/usr/bin/python3

from scipy.interpolate import griddata
import json
import numpy as np

np.set_printoptions(linewidth=200, formatter={'float': lambda x: "{0:0.1f}".format(x)})

afrtable = json.load(open('afrTable1.json'))
vetable = json.load(open('veTable1.json'))

afr_x, afr_y = np.meshgrid(afrtable['x'], afrtable['y'])
points = np.column_stack((afr_x.ravel(), afr_y.ravel()))
values = afrtable['z']

ve_x, ve_y = np.meshgrid(vetable['x'], vetable['y'])

new_afr = griddata(points, values, (ve_x, ve_y), method='linear')
afr_flip = np.flip(np.array(new_afr), 0)

output_file = open('afr.csv', 'w+')
for j in range(16):
    for i in range(16):
        output_file.write(str(round(afr_flip[j][i],1)))
        output_file.write('\t')
    output_file.write('\n')
