import sys, os
import numpy as np
import glob
import pickle
import matplotlib.pyplot as plt
import json
import math

def get_gps(json_path, video_filename):
    with open(json_path) as data_file:
        data = json.load(data_file)

    seg = [x for x in data['segments'] if x['filename'] == video_filename]

    assert (len(seg) == 1)
    seg = seg[0]
    locs = seg['locations']
    loc2nparray = lambda locs, key: np.array([x[key] for x in locs]).ravel()

    res = {}
    bad_video_c = 0
    bad_video_t = 0
    bad_video_same = 0
    for ifile, f in enumerate(locs):
        if int(f['course']) == -1 or int(f['speed']) == -1:
            bad_video_c += 1  # Changed for interpolation
            if bad_video_c >= 3:
                break
        if ifile != 0:
            if int(f['timestamp']) - prev_t > 1100:
                bad_video_t = 1
                break
            if abs(int(f['timestamp']) - int(prev_t)) < 1:
                bad_video_same = 1
                break
        prev_t = f['timestamp']
    if bad_video_c >= 3:
        print('This is a bad video because course or speed is -1', json_path, video_filename)
        return None
    if bad_video_t:
        print('This is a bad video because time sample not uniform', json_path, video_filename)
        return None
    if len(locs) == 0:
        print('This is a bad video because no location data available', json_path, video_filename)
        return None
    if bad_video_same:
        print('This is a bad video because same timestamps', json_path, video_filename)
        return None

    for key in locs[0].keys():
        res[key] = loc2nparray(locs, key)

    # add the starting time point and ending time point as well
    res['startTime'] = seg['startTime']
    res['endTime'] = seg['endTime']

    if res['timestamp'][0] - res['startTime'] > 2000:
        print('This is bad video because starting time too far ahead', json_path, video_filename)
        return None

    if res['endTime'] - res['timestamp'][-1] > 2000:
        print('This is bad video because ending time too far ahead', json_path, video_filename)
        return None

    return res

def get_interp_lat_lon(res, hz):
    querys = np.linspace(res['startTime'], res['endTime'],
                         num=(res['endTime'] - res['startTime'])*hz / 1000)
    latI = np.interp(querys, res['timestamp'], res["latitude"])
    lonI = np.interp(querys, res['timestamp'], res["longitude"])

    # return an N*2 array
    return np.array([latI, lonI]).T

def visLoc(locs, label="NotSet"):
    axis=lambda i: [loc[i] for loc in locs]
    plt.plot(axis(0), axis(1), 'ro')
    #print(axis(0)[204:208])
    #print(axis(1)[204:208])
    #plt.plot(locs[:,0],locs[:,1])
    plt.title("Moving paths from " + label)
    plt.xlabel("West -- East")
    plt.ylabel("South -- North")
    #plt.gca().set_aspect('equal', adjustable='box')
    plt.show()