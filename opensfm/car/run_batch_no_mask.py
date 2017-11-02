import os
import subprocess
import sys

def filter_dirs(dirs):
    ans = []
    for d, d_ori in dirs:
        if '_' in d_ori:
	    continue
        if not os.path.isdir(d_ori):
	    continue
        if not os.path.isdir(d):
            os.mkdir(d)
            subprocess.call(['ln','-s', os.path.join(d_ori,'images'), os.path.join(d,'images') ])
            subprocess.call(['ln','-s', os.path.join(d_ori,'exif'), os.path.join(d,'exif') ])
        if os.path.exists(os.path.join(d, "reconstruction.meshed.json")):
            continue
        ans.append(d)
    return ans

if __name__ == "__main__":
    base_path = "/data/yang/data/opensfm"
    modn = int(sys.argv[1])
    this = int(sys.argv[2])
    if len(sys.argv) >=4 and sys.argv[3].lower() == "run_seg":
        run_seg = True
    else:
        run_seg=False

    gpus = ["6", "7"]

    dirs = []
    
    for dir in os.listdir(base_path):
        dir_nomask = dir + '_nomask'
        full_ori = os.path.join(base_path, dir)
        full = os.path.join(base_path, dir_nomask)
        if not '_' in full_ori:
            dirs.append([full,full_ori])
    
    dirs = sorted(dirs)
    print(dirs)
    dirs = filter_dirs(dirs)
    print(dirs)
    for i, dataset in enumerate(dirs):
        if i%modn == this:
            print dataset
            if run_seg:
                print "running segmentation only"
                e = subprocess.call(["bin/run_seg.sh", dataset, gpus[this % len(gpus)]])
            else:
                e=subprocess.call(["bin/run_nomask", dataset, gpus[this%len(gpus)]])
            print "error code", e
