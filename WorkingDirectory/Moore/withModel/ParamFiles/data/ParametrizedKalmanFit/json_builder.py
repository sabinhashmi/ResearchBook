import json
import numpy as np
import os
'''
Python script to get the json parameter file from the old
.txt and .tab files parameter files.
Execute in this folder will take values from the folder specified in the target_dir variable
lb-conda default should work fine
'''

path = os.getcwd()

target_dir = '24v0'

#read txt file and return a list of lines

data = {}
skip_layer = {'V': [],
              'VUT': [1],
              'UT': [],
              'T': [],
              'TFT': [1],
              'UTTF':[1],
              'UTT_META':[]}

TLay_hardcoded = [[0.0,  0.0874892, -0.0874892, 0.0, 0.0, 0.0874892, -0.0874892, 0.0, 0.0, 0.0874892, -0.0874892, 0.0],
                  [0.0025, 0.0064, 0.0121, 0.0196, 0.0289, 0.04, 0.0529, 0.0676, 0.0841, 0.0, 0.0, 0.0]]

skip_entry = {'V': [0, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
            'VUT': [1, 2, 7, 11, 12, 13, 14, 22, 23, 24, 25, 26, 27, 28, 29, 30],
            'UT': [],
            'T': [],
            'TFT': [1,2,3,4,7,9,10,11,12,13,14,15,16,17,18,19,20],
            'UTTF': [20, 21, 22, 23],
            'UTT_META':[]}

shapes = {'V': (2, 6),
        'VUT': (1, 15),
        'UT': (6, 18),
        'T': (44, 18),
        'TFT': (1, 4),
        'TLayer': (2, 12),
        'UTLayer': (1, 4),
        'UTTF': (1, 20),
        'UTT_META': (1, 19)}

for mag in ['MagDown', 'MagUp']:
    for device in ['V', 'VUT', 'UT', 'T', 'TFT', 'UTTF']:
        file = open(f"{path}/{target_dir}/{mag}/params_predict{device}.txt", "r")
        lines = file.readlines()
        file.close()

        #counter
        layer_counter = 0
        layer_arr_index = 0

        arr = np.ones(shapes[device]) * 123456789 # just let's me spot failed entries faster
        for line in lines:
            if layer_counter in skip_layer[device]:
                layer_counter += 1
                continue
            layer_counter += 1

            entry_counter = 0
            entry_arr_index = 0

            for entry in line.split()[1:]:
                if entry_counter in skip_entry[device]:
                    entry_counter += 1
                    continue
                arr[layer_arr_index][entry_arr_index] = float(entry)
                entry_arr_index += 1
                entry_counter += 1
            layer_arr_index += 1
        data[f'{device}Params_{mag}'] = arr.tolist()

## Get Meta data fro UT->T extrapolation
for mag in ['MagDown', 'MagUp']:
    file = open(f"{path}/{target_dir}/{mag}/params_UTT_v0.tab", "r")
    lines = file.readlines()
    lines = lines[:2]
    file.close()
    # read in the first two lines of the *.tab file
    arr = np.ones(shapes['UTT_META']) * 123456789 # just let's me spot failed entries faster
    entry_counter = 0
    line_counter = 0
    for line in lines:
        for entry in line.split():
            if line_counter == 0:
                arr[0][entry_counter] = float(entry)
            else:
                arr[0][entry_counter] = entry
            entry_counter += 1
        line_counter += 1
    data[f'UTT_META_{mag}'] = arr.tolist()

file = open(f"{path}/{target_dir}/MagDown/params_TLayer.txt", "r")
lines = file.readlines()
file.close()

# print(lines)

arr = np.ones((len(lines) + 2, len(lines[0].split())-1))
for i in range(len(lines)):
    line = lines[i]
    for j in range(1, len(line.split())):
        entry = line.split()[j]
        # print(entry)
        arr[i][j-1] = float(entry)
arr[-2] = TLay_hardcoded[0]
arr[-1] = TLay_hardcoded[1]

data['TLayer'] = arr.tolist()

file = open(f"{path}/{target_dir}/MagDown/params_UTLayer.txt", "r")
lines = file.readlines()
file.close()

arr = np.ones((len(lines), len(lines[0].split())-1)) * 123456789  # just let's me spot failed entries faster
for i in range(len(lines)):
    line = lines[i]
    for j in range(1, len(line.split())):
        entry = line.split()[j]
        arr[i][j-1] = float(entry)
data['UTLayer'] = arr.tolist()

with open(f"{target_dir}/params.json", "w") as file:
    json.dump(data, file, indent=4)