import argparse
import cv2
import os


def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
                help="path to input directory of images")
ap.add_argument("-t", "--threshold", type=float, default=100.0,
                help="focus measures that fall below this value will be considered 'blurry'")
args = vars(ap.parse_args())


if __name__ == "__main__":
    for root, dir, files in os.walk(args["images"]):
        for file in files:
            if "jpg" in file.lower():
                file_full_path = os.path.join(args["images"], file)
                image = cv2.imread(file_full_path)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                fm = variance_of_laplacian(gray)
                print(file, fm)
                if fm < args["threshold"]:
                    print("moving file %s" % file)
                    blurry_path = os.path.join(args["images"], "blurry")
                    if not os.path.exists(blurry_path):
                        os.mkdir(blurry_path)
                    os.rename(file_full_path, os.path.join(blurry_path, file))
