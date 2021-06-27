#!/usr/bin/python3

import sys
from matplotlib import pyplot as plt
from numpy.lib.function_base import average
import numpy as np

np.set_printoptions(linewidth=200)

ndigits = 1
print_array = 'print' in sys.argv
plot_array = 'plot' in sys.argv

rpm_bins = [i*500 for i in range(16)]
rpm_bins[0] = 600  # was 0
rpm_bins[1] = 800  # was 500
map_bins = [i*6+10 for i in range(16)]


def logfct(l):
    AFRcor = l['AFR']*(l['EGO cor1']/100)
    # ratio = (AFRcor/l['AFR Target 1'])-1
    # return ratio * l['VE1']
    # return l['AFR']
    return AFRcor


def parse_file(input_file):
    result = {}
    result['title'] = input_file.readline().strip()
    result['time'] = input_file.readline().strip()
    result['log'] = []
    titles = input_file.readline().strip().split('\t')
    input_file.readline() # units
    for line in input_file:
        items = line.strip().replace(',', '.').split('\t')
        log = {}
        for k, v in enumerate(items):
            log[titles[k]] = float(v)
        # log['AFR'] *= 10
        # log['VE1'] /= 10
        result['log'] += [log]
    return result


def get_points(log, fct):
    x = []
    y = []
    z = []
    ready = False
    fuel_cut = False
    fuel_cut_timeout = 0
    for l in log['log']:
        if not ready:
            ready = l['AFR'] >= 10 and l['Fuel: Warmup cor'] == 100 and l['RPM'] > 500
        elif fuel_cut:
            if l['PW'] == 0 or l['AFR'] > 18:
                fuel_cut_timeout = l['Time']
            else:
                fuel_cut = (l['Time'] - fuel_cut_timeout) < 2
        else:
            fuel_cut = l['TPS'] < 10 and l['PW'] == 0
            x += [l['RPM']]
            y += [l['MAP']]
            z += [fct(l)]
    return x, y, z


def fill_grid(x_bins, y_bins, points):
    x1, y1, z1 = points
    grid_z = [[[] for _ in range(16)] for _ in range(16)]

    for i, _ in enumerate(x1):
        x = x_bins[i]
        y = y_bins[i]
        grid_z[y][x] += [[x1[i], y1[i], z1[i]]]
        if (x > 0):
            grid_z[y][x-1] += [[x1[i], y1[i], z1[i]]]
        if (y > 0):
            grid_z[y-1][x] += [[x1[i], y1[i], z1[i]]]
        if (x > 0) and (y > 0):
            grid_z[y-1][x-1] += [[x1[i], y1[i], z1[i]]]
    return grid_z


def interpolate_grid(grid_z, fill_value):
    def Z(X, Y, params):
        a, b, c, d = params
        return -(a*X + b*Y + d)/c

    grid_c = np.zeros(256)
    grid_s = np.zeros(256)
    for j in range(16):
        for i in range(16):
            if len(grid_z[j][i]) > 10:
                x = np.array(grid_z[j][i])[:, 0]
                y = np.array(grid_z[j][i])[:, 1]
                z = np.array(grid_z[j][i])[:, 2]

                A = np.c_[x, y, np.ones(x.shape)]
                C, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
                vert_params = C[0], C[1], -1., C[2]

                a = Z(rpm_bins[i], map_bins[j], vert_params)
                b = average(z)
                c = abs(a-b)
                d = (a + b*c) / (c+1)
                if abs(a-d) > 1:
                    print(rpm_bins[i], map_bins[j], a, '->', d)
                a = round(d, ndigits)
                grid_z[j][i] = a
                grid_c[i + j * 16] = a
                grid_s[i + j * 16] += len(x)
            else:
                grid_z[j][i] = 0
                grid_c[i + j * 16] = fill_value

    return grid_z, grid_c, grid_s


print(end='.', flush=True)
log = parse_file(open(sys.argv[1], encoding='windows-1252'))
print(end='.', flush=True)
points = get_points(log, logfct)
print(end='.', flush=True)
grid_z = fill_grid(np.digitize(points[0], rpm_bins), np.digitize(
    points[1], map_bins), points)
print(end='.', flush=True)
z, c, s = interpolate_grid(grid_z, average(points[2]))
print('!')

flip_z = np.flip(np.array(grid_z), 0)

if print_array:
    print(flip_z)

if plot_array:
    grid_x, grid_y = np.meshgrid(rpm_bins, map_bins)
    plt.scatter(grid_x, grid_y, c=c, s=s, alpha=0.25, cmap='viridis')
    plt.grid()
    plt.show()

output_file = open("output.csv", "w+")
for j in range(16):
    for i in range(16):
        output_file.write(str(flip_z[j][i]))
        output_file.write('\t')
    output_file.write('\n')