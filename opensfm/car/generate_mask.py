# convert the segmentation to 0/1 mask and generate mask_list
from PIL import Image
import os
import argparse
import subprocess


def convert_mask(value):
    # city scape features here
    # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
    if value in range(10, 19):
        # throw away
        # 10 sky, 11 person, 12 rider, 13 car, 14 truck, 15 bus, 16 train, 17 motorcycle, 18 bicycle.
        return 0
    else:
        return 255


def seg_to_mask(seg_path, mask_path):
    im = Image.open(seg_path)  # Can be many different formats.
    pix = im.load()
    width, height = im.size
    for x in range(width):
        for y in range(height):
            pix[x, y] = convert_mask(pix[x, y])

    im.save(mask_path)  # Save the modified pixels as png

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="dataset dir")
parser.add_argument("--image_height", default=720, help="dataset dir", required=False)
args = parser.parse_args()

if __name__ == "__main__":
    base = args.dataset
    path_images = os.path.join(base, "images")
    path_seg = os.path.join(path_images, "output/results/frontend_vgg")
    path_mask = os.path.join(base, "masks")
    if not os.path.exists(path_mask):
        os.mkdir(path_mask)

    if not os.path.exists(path_seg):
        # segmentation not exist yet, call dilation
        subprocess.call(["opensfm/car/generate_segmentation.sh", path_images, str(args.image_height)])

    # convert segmentation into masks
    for root, dirs, files in os.walk(path_seg):
        for file in files:
            if file.endswith("png"):
                seg_to_mask(os.path.join(path_seg, file),
                            os.path.join(path_mask, file))
