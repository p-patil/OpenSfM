import os
import subprocess
import sys

if __name__ == "__main__":
    base_path = "/data/yang/data/opensfm"

    for dir in os.listdir(base_path):
        full = os.path.join(base_path, dir)
        mask = os.path.join(full, "masks")
        if os.path.exists(mask):
            if len(os.listdir(mask))< 1000:
                print full
