# convert the segmentation to 0/1 mask and generate mask_list
from PIL import Image
import os
import argparse
import subprocess
import numpy as np
#import matplotlib.pyplot as pl
import cv2

def convert_mask(value):
    # city scape features here
    # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
    if value in range(10, 19):
        # throw away
        # 10 sky, 11 person, 12 rider, 13 car, 14 truck, 15 bus, 16 train, 17 motorcycle, 18 bicycle.
        return 0
    else:
        return 255


def seg_to_mask(seg_path, mask_path, hood_height):
    im = Image.open(seg_path)  # Can be many different formats.
    pix = np.array(im)
    pix = (pix < 10) * 255
    pix[-hood_height:, :] = 0

    cv2.imwrite(mask_path, pix.astype(np.uint8))

'''
def hood_mask(path_images, downsample, std_thres):
    ifile = 0
    images = []
    for item in sorted(os.listdir(path_images)):
        full_path = os.path.join(path_images, item)
        fname = item.lower()
        if os.path.isfile(full_path) and (fname.endswith("jpg") or fname.endswith("jpeg") or fname.endswith("png")):
            if ifile % downsample == 0:
                print(item)
                im = Image.open(os.path.join(path_images, item))  # Can be many different formats.
                pix = np.array(im)
                if False:
                    pix = cv2.cvtColor(pix, cv2.COLOR_RGB2HSV)

                if False:
                    print(type(pix[0,0,0]))
                    pix[:,:,2] = 128
                    pix = cv2.cvtColor(pix, cv2.COLOR_HSV2RGB)
                    pl.imshow(pix)
                    pl.show()
                    return
                if False:
                    pix = pix[:, :, 0:2]

                pix = pix.mean(axis=2)
                images.append(pix)
            ifile += 1

    all = np.stack(images, axis=0)
    print(all.shape)
    std = all.std(axis=0)
    print(std.shape)
    #pl.imshow(std<std_thres)
    pl.imshow(std)
    std = np.minimum(std*4, 255)
    stdim = Image.fromarray(std.astype(np.uint8))
    stdim.save("/Users/yang/Downloads/std.jpg")
    pl.show()

def hood_mask_edge(path_images, downsample, std_thres):
    ifile = 0
    images = []
    for item in sorted(os.listdir(path_images)):
        full_path = os.path.join(path_images, item)
        fname = item.lower()
        if os.path.isfile(full_path) and (fname.endswith("jpg") or fname.endswith("jpeg") or fname.endswith("png")):
            if ifile % downsample == 0:
                print(item)
                im = Image.open(os.path.join(path_images, item))  # Can be many different formats.

                im=Image.open("/Users/yang/Downloads/std.jpg")
                pix = np.array(im)
                edges = cv2.Canny(pix, 100, 200)
                pl.imshow(edges)
                pl.show()
                return

                pix = pix.mean(axis=2)
                images.append(pix)
            ifile += 1

    all = np.stack(images, axis=0)
    print(all.shape)
    pl.show()
'''

parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="dataset dir")
parser.add_argument("--image_height", default=720, help="dataset dir", required=False)
'''
parser.add_argument("--hood_downsample",
                    default=10,
                    help="when trying to detect car hood, what downsample ratio",
                    required=False)
parser.add_argument("--hood_std",
                    default=20.0,
                    help="the std for detecting car hood",
                    required=False)
'''
parser.add_argument("--hood_height",
                    default=150,
                    help="how many pixels to remove for hood",
                    required=False)
parser.add_argument("--seg_relative_path",
                    default="output/results/joint",
                    help="the path of segmentations relative to images",
                    required=False)
parser.add_argument("--gpu",
                    default="0",
                    help="which GPU to use to run Dilation",
                    required=False)
args = parser.parse_args()

if __name__ == "__main__":
    base = args.dataset
    path_images = os.path.join(base, "images")
    path_seg = os.path.join(path_images, args.seg_relative_path)
    path_mask = os.path.join(base, "masks")

    if not os.path.exists(path_mask):
        if not os.path.exists(path_seg):
            # segmentation not exist yet, call dilation
            e = subprocess.call(
                ["opensfm/car/generate_segmentation.sh", path_images, str(args.image_height), str(args.gpu)])
            if e:
                print "some error happend when calling the segmentation generation"
                exit(0)

        os.mkdir(path_mask)
    else:
        print("mask has already been generated, exit")
        exit()

    # convert segmentation into masks
    for item in sorted(os.listdir(path_seg)):
        if item.lower().endswith("png"):
            seg_to_mask(os.path.join(path_seg, item),
                        os.path.join(path_mask, item),
                        int(args.hood_height))
