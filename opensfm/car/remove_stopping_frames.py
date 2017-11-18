#!/usr/bin/env python
import sys
sys.path.append("/home/piyush/Academics/Berkeley/deepdrive/OpenSfM")
import numpy as np
from PIL import Image 
from opensfm import dataset
import os
import argparse
from opensfm import matching
import cv2
import subprocess
import shutil
from pathlib2 import Path

def homography_inlier_ratio(p1, p2, matches, args):
    # test whether this pair forms a homography
    p1 = p1[:, 0:2]
    p2 = p2[:, 0:2]
    p1matched = p1[matches[:, 0], :]
    p2matched = p2[matches[:, 1], :]
    # use a stricter threshold
    H, inliers = cv2.findHomography(p1matched, p2matched, cv2.RANSAC,
                                    float(args.homography_ransac_threshold))
    inliers_ratio = inliers.sum() * 1.0 / matches.shape[0]

    return inliers_ratio

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='dataset to process')
    parser.add_argument('--homography_ransac_threshold',
                        help='the threshold used to match homography',
                        default=0.004)
    parser.add_argument('--homography_inlier_ratio',
                        help='the lower bound of homography inlier ratio to be considered as the same frame',
                        default=0.90)
    parser.add_argument('--matching_mod',
                        help='could either be good or fast',
                        default="good")

    args = parser.parse_args()

    is_good = (args.matching_mod == "good")
    data = dataset.DataSet(args.dataset)
    images = sorted(data.images())
    config = data.config

    # the current image, next image is used as potentials to be the same as this image
    im1i = 0
    retained = [images[0]]
    indexes = [0]

    if is_good:
        robust_matching_min_match = config['robust_matching_min_match']
        cameras = data.load_camera_models()
        exifs = {im: data.load_exif(im) for im in images}

        while im1i + 1 < len(images):
            im1 = images[im1i]
            # while the next image exists
            p1, f1, c1 = data.load_features(im1)
            i1 = data.load_feature_index(im1, f1)
            # get the cached features
            if data.matches_exists(im1):
                im1_matches = data.load_matches(im1)
            else:
                im1_matches = {}
            modified = False
            
            # Include segmentations
            path_seg = data.data_path + "/images/output/results/frontend_vgg/" + os.path.splitext(im1)[0]+'.png'    
            file_name = Path(path_seg)
            if file_name.is_file():
                im1_seg = Image.open(path_seg)
                im1_seg = np.array(im1_seg)
            
                idx_u1 = im1_seg.shape[1]*(p1[:,0] + 0.5)
                idx_v1 = im1_seg.shape[0]*(p1[:,1] + 0.5)
                im1_seg = im1_seg[idx_v1.astype(np.int),idx_u1.astype(np.int)]
            else:
                im1_seg = None

            for im2i in range(im1i+1, len(images)):
                # match this image against the inow
                im2 = images[im2i]
                p2, f2, c2 = data.load_features(im2)

                path_seg = data.data_path + "/images/output/results/frontend_vgg/" + os.path.splitext(im2)[0]+'.png'    
                file_name = Path(path_seg)
                if file_name.is_file():
                    im2_seg = Image.open(path_seg)
                    im2_seg = np.array(im2_seg)
                else:
                    im2_seg = None
                if im2 not in im1_matches:
                    modified = True
                    i2 = data.load_feature_index(im2, f2)
                    
                    if file_name.is_file():
                        idx_u2 = im2_seg.shape[1]*(p2[:,0]+0.5)
                        idx_v2 = im2_seg.shape[0]*(p2[:,1]+0.5)
                        im2_seg = im2_seg[idx_v2.astype(np.int),idx_u2.astype(np.int)]
                    else:
                        ims2_seg = None
                    matches = matching.match_symmetric(f1, i1, f2, i2, config,
                                                      im1_seg, im2_seg)

                    if len(matches) < robust_matching_min_match:
                        # this image doesn't have enough matches with the first one
                        # i.e. either of them is broken, to be safe throw away both
                        print("%s and %s don't have enough matches, skipping" % (im1, im2))
                        im1i = im2i + 1
                        break

                    # robust matching
                    camera1 = cameras[exifs[im1]['camera']]
                    camera2 = cameras[exifs[im2]['camera']]

                    rmatches = matching.robust_match(p1, p2, camera1, camera2, matches,
                                                     config)
                    if len(rmatches) < robust_matching_min_match:
                        im1_matches[im2] = []
                    else:
                        im1_matches[im2] = rmatches
                    #print("computed match between %s and %s" % (im1, im2))
                else:
                    rmatches = im1_matches[im2]

                if len(rmatches) < robust_matching_min_match:
                    print("%s and %s don't have enough robust matches, skipping" % (im1, im2))
                    im1i = im2i + 1
                    break

                inliers_ratio = homography_inlier_ratio(p1, p2, rmatches, args)
                print("im %s and im %s, homography ratio is %f" % (im1, im2, inliers_ratio))
                if inliers_ratio <= float(args.homography_inlier_ratio):
                    # this figure considered as not the same
                    retained.append(im2)
                    indexes.append(im2i)
                    im1i = im2i
                    break
                else:
                    print("throw away %s" % im2)
            else:
                im1i += 1

            if modified:
                data.save_matches(im1, im1_matches)
    else:
        # we should run neighbourhood matching anyway
        # make a copy of the old config
        config_path = os.path.join(data.data_path, "config.yaml")
        config_bak  = config_path + ".bak"
        os.rename(config_path, config_bak)

        # replace the line with neighbour 2
        subprocess.call(['sed -e "s/matching_order_neighbors:.*/matching_order_neighbors: 2/" ' +
                         config_bak + ' > ' + config_path], shell=True)

        subprocess.call(["bin/opensfm", "match_features", args.dataset])

        # remove the replaced file
        os.remove(config_path)
        # move back
        os.rename(config_bak, config_path)

        # using the loaded features after ransac
        # slightly different logic here, we use the nearby frames' matches only
        for i1, im1 in enumerate(images):
            im1_matches = data.load_matches(im1)
            p1, f1, c1 = data.load_features(im1)

            if i1+1 < len(images):
                im2 = images[i1+1]
                p2, f2, c2 = data.load_features(im2)
                match = im1_matches[im2]
                if match == []:
                    print("im %s and im %s don't have match, throw away 2nd" % (im1, im2))
                    continue
                # match is a list of tuples indicating which feature do I use for 2 images
                inliers_ratio = homography_inlier_ratio(p1, p2, match, args)
                print("im %s and im %s, homography ratio is %f" % (im1, im2, inliers_ratio))
                if inliers_ratio <= float(args.homography_inlier_ratio):
                    retained.append(im2)
                    indexes.append(i1+1)
                else:
                    print("throw away %s" % im2)

    # TODO: investigate whether need to remove further stop frames
    '''
    # refine the list of remaining images by removing the isolated frames
    refined = [retained[0]]
    nn = 3
    for i in range(1, len(retained)-1):
        if abs(indexes[i]-indexes[i-1])<=nn or abs(indexes[i]-indexes[i+1])<=nn:
            refined.append(retained[i])
    refined.append(retained[-1])
    retained = refined
    '''

    # overwrite the image list if it exists
    image_list = os.path.join(data.data_path, "image_list.txt")
    with open(image_list, "w") as f:
        for im in retained:
            f.write("images/"+im+"\n")

if __name__ == "__main__":
    main()
