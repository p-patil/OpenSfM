import os
import subprocess
import sys
from parse_ride_json import generate_gps_figure

if __name__ == "__main__":
    base_path = "/data/yang/data/opensfm"

    for dir in os.listdir(base_path):
        full = os.path.join(base_path, dir)
        mask = os.path.join(full, "masks")
        if os.path.exists(mask):
            if len(os.listdir(mask))< 1000:
                print full, "bad mask"
        if os.path.exists(os.path.join(full, "reconstruction.meshed.json")):
            print "finished", dir
        generate_gps_figure(os.path.join(full, "ride.json"),
                            dir+".mov",
                            full)
