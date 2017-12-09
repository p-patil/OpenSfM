import sys, os
sys.path.append("/home/piyush/Academics/Berkeley/deepdrive/mapping-dev/alpha/reconstruction/OpenSfM")
from opensfm.car import parse_ride_json
from opensfm.pexif import JpegFile

if __name__ == "__main__":
    print "attempting to add gps to images"

    dataset_path = sys.argv[1]

    json_path = os.path.join(dataset_path, "ride.json")
    if os.path.exists(json_path):
        print "found ride.json, adding gps to images"

        if dataset_path[-1] == "/":
            dataset_path = dataset_path[:-1]

        head, tail = os.path.split(dataset_path)

        # purge the _xxx component
        tail = tail.split("_")[0]

        gps_res = parse_ride_json.get_gps(json_path, tail + ".mov")
        gps_interp = parse_ride_json.get_interp_lat_lon(gps_res, 30)

        # get all the files
        images = []
        imbase = os.path.join(dataset_path, "images")
        for im in os.listdir(imbase):
            imlow = im.lower()
            if ("jpg" in imlow) or ("png" in imlow) or ("jpeg" in imlow):
                images.append(im)
        images = sorted(images)

        if len(images) > gps_interp.shape[0]:
            print "length of gps insufficient, exit"
            exit(0)
        else:
            for i, shortname in enumerate(images):
                imname = os.path.join(imbase, shortname)
                ef = JpegFile.fromFile(imname)
                ef.set_geo(gps_interp[i, 0], gps_interp[i, 1])
                ef.writeFile(imname)
    else:
        print "json not found"

    print "exit"
    print
