import matplotlib as mpl
mpl.use('Agg')
import os
import subprocess
import sys
from parse_ride_json import generate_gps_figure

if __name__ == "__main__":
    base_path = sys.argv[1]
    #base_path = "/data/yang/data/opensfm"

    for dir in os.listdir(base_path):
        full = os.path.join(base_path, dir)
        if not os.path.isdir(full):
            continue
        
        mask = os.path.join(full, "masks")
        if os.path.exists(mask):
            if len(os.listdir(mask))< 1000:
                print full, "bad mask"
        if os.path.exists(os.path.join(full, "reconstruction.meshed.json")):
            print "finished", dir

        json_path = os.path.join(full, "ride.json")
        if os.path.exists(json_path):
            generate_gps_figure(json_path,
                                dir+".mov",
                                full)
