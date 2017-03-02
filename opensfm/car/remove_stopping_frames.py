from opensfm import dataset
import os
import argparse
from opensfm import matching
import cv2

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='dataset to process')
    parser.add_argument('--homography_ransac_threshold',
                        help='the threshold used to match homography',
                        default=0.002)
    parser.add_argument('--homography_inlier_ratio',
                        help='the lower bound of homography inlier ratio to be considered as the same frame',
                        default=0.95)
    parser.add_argument('--matching_mod',
                        help='could either be cached_ransac or computed_non_ransac',
                        default="cached_ransac")

    args = parser.parse_args()

    is_ransac = (args.matching_mod == "cached_ransac")
    data = dataset.DataSet(args.dataset)
    images = sorted(data.images())
    config = data.config

    # first check whether we should run this program
    if not os.path.exists(os.path.join(data.data_path, "matches")):
        print("run the matching first, then remove stopping frames")
        exit()

    if os.path.exists(os.path.join(data.data_path, "matches_with_stop")):
        print("stopping frames has been removed, skpping")
        exit()

    # the current image, next image is used as potentials to be the same as this image
    im1i = 0
    retained = [images[0]]

    if not is_ransac:
        while im1i + 1 < len(images):
            im1 = images[im1i]
            # while the next image exists
            p1, f1, c1 = data.load_features(im1)
            i1 = data.load_feature_index(im1, f1)

            for im2i in range(im1i+1, len(images)):
                # match this image against the inow
                im2 = images[im2i]
                p2, f2, c2 = data.load_features(im2)

                i2 = data.load_feature_index(im2, f2)

                matches = matching.match_symmetric(f1, i1, f2, i2, config)
                # TODO: potentially there is so few matches that should be handled with special case

                inliers_ratio = homography_inlier_ratio(p1, p2, matches, args)
                print("im %s and im %s, homography ratio is %f" % (im1, im2, inliers_ratio))
                if inliers_ratio <= float(args.homography_inlier_ratio):
                    # this figure considered as not the same
                    retained.append(im2)
                    im1i = im2i
                    break
    else:
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
                else:
                    print("throw away %s" % im2)


    # overwrite the image list if it exists
    image_list = os.path.join(data.data_path, "image_list.txt")
    if os.path.exists(image_list):
        os.rename(image_list,
                  os.path.join(data.data_path, "image_list.txt.with_stop"))
    with open(image_list, "w") as f:
        for im in retained:
            f.write("images/"+im+"\n")

    # move away the old matches
    os.rename(os.path.join(data.data_path, "matches"),
              os.path.join(data.data_path, "matches_with_stop"))
