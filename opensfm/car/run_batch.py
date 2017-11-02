import os
import subprocess
import sys
import eval
import pickle

def filter_dirs(dirs, program):
    ans = []
    for d in dirs:
        if "_" in d:
            continue
        if not os.path.isdir(d):
            continue
        if program == "opensfm":
            if os.path.exists(os.path.join(d, "reconstruction.meshed.json")):
                continue
        elif program == "dso" or program == "orb":
            if os.path.exists(os.path.join(d, program+".txt")):
                with open(os.path.join(d, program+".txt"), "r") as f:
                    line = f.readline()
                if line.strip() != "error":
                    continue

        ans.append(d)
    return ans

if __name__ == "__main__":
    base_path = "/data/yang/data/opensfm"
    modn = int(sys.argv[1])
    this = int(sys.argv[2])

    program = sys.argv[3].lower()

    gpus = ["6", "7"]

    dirs = []
    for dir in os.listdir(base_path):
        full = os.path.join(base_path, dir)
        dirs.append(full)

    dirs = sorted(dirs)
    dirs = filter_dirs(dirs, program)

    results = {}
    if program == "eval_all" and os.path.exists("evaluated.pkl"):
        with open("evaluated.pkl", "rb") as f:
            results = pickle.load(f)

    for i, dataset in enumerate(dirs):
        if i%modn == this:
            print dataset

            if program == "seg":
                e = subprocess.call(["bin/run_seg.sh", dataset, gpus[this % len(gpus)]])
            elif program == "opensfm":
                e=subprocess.call(["bin/run_all", dataset, gpus[this%len(gpus)]])
            elif program == "dso" or program == "dso_mask":
                # generate the time stamps
                with open(os.path.join(dataset, "images") + "/times.txt", "w") as f0:
                    largest = -1
                    for f in os.listdir(os.path.join(dataset, "images")):
                        if f.endswith("jpg"):
                            id = int(f.split(".")[0])
                            if id>largest:
                                largest = id
                    for i in range(largest):
                        f0.write(str(i)+ " "+str(i*1.0/30)+"\n")

                mask_flag = "mask=../masks/" if "mask" in program else ""
                if len(sys.argv)>=5:
                    relative = sys.argv[4]
                else:
                    relative = "dso/"

                e=subprocess.call([ relative+"build/bin/dso_dataset",
                                    "files="+dataset+"/images/",
                                    "calib="+relative+"camera_nexar.txt",
                                    "preset=2",
                                    "mode=1",
                                    "nogui=1",
                                    "nolog=1",
                                    mask_flag])
                if not os.path.exists("result.txt"):
                    subprocess.call(["echo 'error' > result.txt "], shell=True)
                os.rename("result.txt", os.path.join(dataset, program+".txt"))
            elif program == "orb" or program == "orb_mask":
                if len(sys.argv)>=5:
                    relative = sys.argv[4]
                else:
                    relative = "ORB_SLAM2/"

                mask_flag = "../masks/" if "mask" in program else ""
                e = subprocess.call([relative + "Examples/Monocular/mono_nexar",
                                     relative + "Vocabulary/ORBvoc.txt",
                                     relative + "Examples/Monocular/nexar.yaml",
                                     dataset + "/images",
                                     "0",
                                     mask_flag])
                outname = "KeyFrameTrajectory.txt"
                if not os.path.exists(outname):
                    subprocess.call(["echo 'error' > " + outname], shell=True)
                os.rename(outname, os.path.join(dataset, program+".txt"))
            elif program == "eval_all":
                if dataset not in results:
                    results[dataset] = {}

                for method in ["opensfm", "orb", "dso"]:
                    if method not in results[dataset]:
                        results[dataset][method] = {}

                    for mask in ["true", "false"]:
                        print(method, mask)

                        if mask not in results[dataset][method]:
                            results[dataset][method][mask] = eval.eval_api(method, dataset, mask)
                            print(results[dataset][method][mask])
                        else:
                            print("cached", results[dataset][method][mask])

                        with open("evaluated.pkl", "wb") as f:
                            pickle.dump(results, f)
                e=0
            else:
                raise ValueError("invalid program parameter")

            print "error code", e
