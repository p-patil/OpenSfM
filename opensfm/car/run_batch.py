import os
import subprocess
import sys

if __name__ == "__main__":
    base_path = "/data/yang/data/opensfm"
    modn = int(sys.argv[1])
    this = int(sys.argv[2])
    if len(sys.argv) >=4:
        if sys.argv[3].lower() == "run_seg":
            run_seg = True
    else:
        run_seg=False

    gpus = ["0", "1", "2", "3"]

    dirs = []
    for dir in os.listdir(base_path):
        full = os.path.join(base_path, dir)
        dirs.append(full)

    dirs = sorted(dirs)

    for i, dataset in enumerate(dirs):
        if i%modn == this:
            print dataset
            if run_seg:
                print "running segmentation only"
                e = subprocess.call(["bin/run_seg.sh", dataset, gpus[this % len(gpus)]])
            else:
                e=subprocess.call(["bin/run_all", dataset, gpus[this%len(gpus)]])
            print "error code", e
