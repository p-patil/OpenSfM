#!/usr/bin/env python
import sys, os, argparse, cv2, subprocess, shutil, time
sys.path.append("/root/deepdrive/OpenSfM")
import numpy as np
from PIL import Image
from opensfm import dataset, matching
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

def get_segmentations(data, im, p, round = False):
    path_seg = data.data_path + "/images/output/results/frontend_vgg/" + os.path.splitext(im)[0]+".png"
    file_name = Path(path_seg)
    if file_name.is_file():
        im_seg = Image.open(path_seg)
        im_seg = np.array(im_seg)

        if round:
            idx_u = im_seg.shape[1] * (p[:, 0] + 0.5)
            idx_v = im_seg.shape[0] * (p[:, 1] + 0.5)
            im_seg = im_seg[idx_v.astype(np.int), idx_u.astype(np.int)]
    else:
        im_seg = None

    return im_seg

# TODO align print statements
def remove_stopping_frames_not_good(args):
    data = dataset.DataSet(args.dataset)

    # The current image, next image is used as potentials to be the same as this image
    images = sorted(data.images())
    retained = [images[0]]
    indexes = [0]

    # We should run neighbourhood matching anyway, first backup the existing config
    config_path = os.path.join(data.data_path, "config.yaml")
    config_bak  = config_path + ".bak"
    os.rename(config_path, config_bak)

    # Replace the line with neighbour 2
    subprocess.call( \
        ["sed -e \"s/matching_order_neighbors:.*/matching_order_neighbors: 2/\" " + config_bak + " > " + config_path], \
        shell=True)


    subprocess.call(["bin/opensfm", "match_features", args.dataset])

    # Move back
    os.remove(config_path)
    os.rename(config_bak, config_path)

    # using the loaded features after ransac
    # slightly different logic here, we use the nearby frames' matches only
    for i1, im1 in enumerate(images):
        im1_matches = data.load_matches(im1)
        p1, f1, c1 = data.load_features(im1)

        if i1 + 1 < len(images):
            im2 = images[i1 + 1]
            p2, f2, c2 = data.load_features(im2)
            match = im1_matches[im2]

            if match == []:
                print("im %s and im %s don't have match, throw away 2nd" % (im1, im2))
                continue

            # match is a list of tuples indicating which feature do I use for 2 images
            inliers_ratio = homography_inlier_ratio(p1, p2, match, args)
            print("computed match between im %s and im %s, homography ratio is %f" % (im1, im2, inliers_ratio))
            if inliers_ratio <= float(args.homography_inlier_ratio):
                retained.append(im2)
                indexes.append(i1 + 1)
            else:
                print("homography inlier ratio is too high, throwing away %s" % im2)

    return retained, indexes

def get_cache(matches_path):
    cache_path = os.path.join(os.path.dirname(matches_path), "matches_cache.txt")

    computed = os.path.exists(cache_path)
    if computed:
        print("found cache of matched images")
        with open(cache_path, "r") as f:
            computed_matches = set([line.strip() for line in f.readlines() if len(line) > 0])

        print("augmenting cache by reading matches directory")
        for matches_file in os.listdir(matches_path):
            image1 = matches_file[: - len("_matches.pkl.gz")]

            import gzip, pickle
            with gzip.open(os.path.join(matches_path, matches_file), "rb") as f:
                matches = pickle.load(f)
                for image2 in matches.keys():
                    computed_matches.add("%s,%s" % (image1, image2))
        print("found %i entries in cache" % len(computed_matches))
    else:
        computed_matches = None

    return cache_path, computed, computed_matches

def process_images(args, data, images, config):
    # TODO remove. Timing every aspect of matching between two images.
    cameras = data.load_camera_models()
    exifs = {im: data.load_exif(im) for im in images}
    im1, im2 = images[0], images[1]
    p1, f1, c1 = data.load_features(im1)
    p2, f2, c2 = data.load_features(im2)
    i1, i2 = data.load_feature_index(im1, f1), data.load_feature_index(im2, f2)
    im1_matches = data.load_matches(im1) if data.matches_exists(im1) else {}
    im1_seg = get_segmentations(data, im1, p1, round = True)
    im2_seg = get_segmentations(data, im2, p2, round = im2 not in im1_matches)

    start = time.time()
    print("%s seconds elapsed" % str(time.time() - start))
    start = time.time()
    print("Computing matching...")
    matches = matching.match_symmetric(f1, i1, f2, i2, config, im1_seg, im2_seg)
    print("%s seconds elapsed" % str(time.time() - start))
    start = time.time()
    print("Computing robust matches")
    # Get camera parameters for robust matching
    camera1, camera2 = cameras[exifs[im1]["camera"]], cameras[exifs[im2]["camera"]]
    rmatches = matching.robust_match(p1, p2, camera1, camera2, matches, config)
    print("%s seconds elapsed" % str(time.time() - start))
    print("Computing homography ratio")
    start = time.time()
    inliers_ratio = homography_inlier_ratio(p1, p2, rmatches, args)
    print("%s seconds elapsed" % str(time.time() - start))
    sys.exit()

    # Iterate over all images
    im1i = 0
    while im1i + 1 < len(images):
        im1 = images[im1i]
        p1, f1, c1 = data.load_features(im1)
        i1 = data.load_feature_index(im1, f1)

        print("processing image %s" % im1)

        # Get cached matches
        im1_matches = data.load_matches(im1) if data.matches_exists(im1) else {}

        # For image 1, compute pairwise matches against all other images and retain the
        # images that are sufficiently different (i.e. have enough robust matches and
        # have a low inliers ratio) from image 1. We essentially skip images until 
        modified = False
        for im2i in range(im1i + 1, len(images)):
            im2 = images[im2i]
            p2, f2, c2 = data.load_features(im2)
            i2 = data.load_feature_index(im2, f2)

            print("\tmatching %s against %s " % (im1, im2))

            # Compute matches between image 1 and image 2 if not already computed
            if im2 not in im1_matches:
                modified = True

                # Include segmentations
                im1_seg = get_segmentations(data, im1, p1, round = True)
                im2_seg = get_segmentations(data, im2, p2, round = im2 not in im1_matches)

                # Compute symmetric matches
                sys.stdout.write("\t\t") # Prepend tabs in prints of match_symmetric
                matches = matching.match_symmetric(f1, i1, f2, i2, config, im1_seg, im2_seg)

                # Not enough matches, so there definitely won't be enough robust matches; all
                # images up to image 2 have already been skipped, so don't compare against them
                if len(matches) < robust_matching_min_match:
                    print("\t\tNot enough symmetric matches, throwing away")
                    im1i = im2i + 1
                    break

                # Get camera parameters for robust matching
                camera1 = cameras[exifs[im1]["camera"]]
                camera2 = cameras[exifs[im2]["camera"]]

                # Compute robust matches, filtering from symmetric matches. Image 2 matches with
                # image 1 only if there are enough matches.
                rmatches = matching.robust_match(p1, p2, camera1, camera2, matches, config)
                im1_matches[im2] = [] if len(rmatches) < robust_matching_min_match else rmatches
            else:
                rmatches = im1_matches[im2]

            # If there aren't enough robust matches, skip
            if len(rmatches) < robust_matching_min_match:
                print("\t\tNot enough robust matches, throwing away")
                im1i = im2i + 1 # Skipping image 2, so don't compare any of the images against it
                break

            # There are enough robust matches between the images, so retain image 2 only if it's
            # sufficiently different from image 1
            inliers_ratio = homography_inlier_ratio(p1, p2, rmatches, args)
            print("\t\tcomputed match between im %s and im %s, homography ratio is %f" % (im1, im2, inliers_ratio))
            if inliers_ratio <= float(args.homography_inlier_ratio):
                print("\t\tImage %s is sufficiently different, retaining" % im2)
                retained.append(im2)
                indexes.append(im2i)
                im1i = im2i # All images up to image 2 have already been skipped, so resume at image 2
                break
            else:
                print("\thomography inlier ratio is too high, throwing away %s" % im2)
        else:
            # Image 2 has enough robust matches but is too similar to image 1, so throw away image 1
            im1i += 1

        # If image 1's matches were updated, re-save to file (overwriting old matches file)
        if modified:
            data.save_matches(im1, im1_matches)

    return retained, indexes

def remove_stopping_frames_good(args):
    data = dataset.DataSet(args.dataset)
    config = data.config

    # Check which, if any, matches have already been computed
    # cache_path, computed, computed_matches = get_cache(data.matches_path())

    # The current image, next image is used as potentials to be the same as this image
    images = sorted(data.images())
    retained = [images[0]]
    indexes = [0]

    robust_matching_min_match = config["robust_matching_min_match"]
    cameras = data.load_camera_models()
    exifs = {im: data.load_exif(im) for im in images}

    print("computing matches")

    process_images(args, data, images, config)

    # im1i = 0
    # while im1i + 1 < len(images):
        # im1 = images[im1i]

        # print("processing image %s" % im1)

        # p1, f1, c1 = data.load_features(im1)
        # i1 = data.load_feature_index(im1, f1)

        # # Get the cached features
        # if data.matches_exists(im1):
            # im1_matches = data.load_matches(im1)
        # else:
            # im1_matches = {}

        # # Match against all subsequent images
        # modified = False
        # for im2i in range(im1i + 1, len(images)):
            # im2 = images[im2i]

            # print("\tmatching %s against %s " % (im1, im2))

            # # # Check if already computed, and if not, mark as computed
            # # if computed and "%s,%s" % (im1, im2) in computed_matches:
                # # print("\t\tcache hit")
                # # continue
            # # else:
                # # print("\t\twriting to cache")
                # # with open(cache_path, "a") as f:
                    # # f.write("%s,%s\n" % (im1, im2))

            # p2, f2, c2 = data.load_features(im2)

            # if im2 not in im1_matches:
                # modified = True
                # i2 = data.load_feature_index(im2, f2)

                # # Include segmentations
                # im1_seg = get_segmentations(data, im1, p1, round = True)
                # im2_seg = get_segmentations(data, im2, p2, round = im2 not in im1_matches)

                # sys.stdout.write("\t\t") # Prepend tabs in prints of match_symmetric
                # matches = matching.match_symmetric(f1, i1, f2, i2, config, im1_seg, im2_seg)

                # if len(matches) < robust_matching_min_match:
                    # # This image doesn't have enough matches with the first one i.e. either of
                    # # them is broken; to be safe throw away both
                    # print("\t%s and %s don't have enough matches, skipping" % (im1, im2))
                    # im1i = im2i + 1
                    # break

                # # Robust matching
                # camera1 = cameras[exifs[im1]["camera"]]
                # camera2 = cameras[exifs[im2]["camera"]]

                # rmatches = matching.robust_match(p1, p2, camera1, camera2, matches, config)
                # if len(rmatches) < robust_matching_min_match:
                    # im1_matches[im2] = []
                # else:
                    # im1_matches[im2] = rmatches
            # else:
                # rmatches = im1_matches[im2]

            # if len(rmatches) < robust_matching_min_match:
                # print("\t%s and %s don't have enough robust matches, skipping" % (im1, im2))
                # im1i = im2i + 1
                # break

            # inliers_ratio = homography_inlier_ratio(p1, p2, rmatches, args)
            # print("\t\tcomputed match between im %s and im %s, homography ratio is %f" % (im1, im2, inliers_ratio))
            # if inliers_ratio <= float(args.homography_inlier_ratio):
                # # this figure considered as not the same
                # retained.append(im2)
                # indexes.append(im2i)
                # im1i = im2i
                # break
            # else:
                # print("\thomography inlier ratio is too high, throwing away %s" % im2)
        # else:
            # im1i += 1

        # if modified:
            # data.save_matches(im1, im1_matches)

    # return retained, indexes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="dataset to process")
    parser.add_argument("--homography_ransac_threshold",
                        help="the threshold used to match homography",
                        default=0.004)
    parser.add_argument("--homography_inlier_ratio",
                        help="the lower bound of homography inlier ratio to be considered as the same frame",
                        default=0.90)
    parser.add_argument("--matching_mod",
                        help="could either be good or fast",
                        default="good")

    print("removing stopping frames")

    args = parser.parse_args()
    data = dataset.DataSet(args.dataset)

    start = time.time()

    is_good = (args.matching_mod == "good")
    if is_good:
        retained, indexes = remove_stopping_frames_good(args)
    else:
        retained, indexes = remove_stopping_frames_not_good(args)

    end = time.time()

    print("removing stopping frames took %s seconds" % str(end - start))

    # Overwrite the image list if it exists
    image_list = os.path.join(data.data_path, "image_list.txt")
    with open(image_list, "w") as f:
        for im in retained:
            f.write("images/" + im + "\n")

    print("exit\n")

if __name__ == "__main__":
    main()
