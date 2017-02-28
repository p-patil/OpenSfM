import os
from opensfm import features
from opensfm import dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as pl
from opensfm import matching
import imp
import cv2

plot_matches = imp.load_source('plot_matches', 'bin/plot_matches')
plot_features = imp.load_source('plot_features', 'bin/plot_features')

def filter_by_segmentation(points, imname, config, data):
    # return a 0-1 array of whehter  this point belongs to the mask
    homography_seg_relative_path=config.get("homography_seg_relative_path",
                                            "output/results/frontend_vgg/")
    image_path = data.image_files[imname]
    head, tail = os.path.split(image_path)
    seg_path = os.path.join(head, homography_seg_relative_path, imname.split('.')[0] + ".png")

    im = Image.open(seg_path)

    if False:
        # debug
        b=Image.fromarray(np.array(im)*10)
        b.show()

    pix = im.load()
    width, height = im.size
    points = points[:, 0:2]
    points = features.denormalized_image_coordinates(points, width, height)

    ans = []
    for p in points:
        if pix[int(p[0]), int(p[1])] == 0:
            ans.append(True)
        else:
            ans.append(False)
    return ans

def filter_by_segmentation_test(im1, config, data):
    p1, f1, c1 = data.load_features(im1)

    mask = filter_by_segmentation(p1, im1, config, data)
    plot_features.plot_features(data.image_as_array(im1), p1, c1, False)
    pl.show()
    plot_features.plot_features(data.image_as_array(im1), p1[mask, :], c1, False)
    pl.show()

def load_masked(im1, data, config):
    p1, f1, c1 = data.load_features(im1)
    p1 = p1[:, 0:2]

    mask1 = filter_by_segmentation(p1, im1, config, data)
    p1 = p1[mask1, :]
    f1 = f1[mask1]
    i1 = features.build_flann_index(f1, data.config)

    return p1, f1, i1


def match_lowe(index, f2, config, reverse):
    knn = 1

    search_params = dict(checks=config.get('flann_checks', 200))
    results, dists = index.knnSearch(f2, knn, params=search_params)

    ans = []
    for i in range(len(results)):
        for nn in range(knn):
            if not reverse:
                match = (results[i][nn], i)
            else:
                match = (i, results[i][nn])
            ans.append(match)

    return ans


def homography_match(fi, indexi, fj, indexj, config):
    matches_ij = match_lowe(indexi, fj, config, False)
    matches_ji = match_lowe(indexj, fi, config, True)

    matches = set(matches_ij).intersection(set(matches_ji))
    return np.array(list(matches), dtype=int)


def homography_match_test(matches, data, p1, p2,
                          random_downsample_keep = 0.1):
    print("found %d matches" % matches.shape[0])
    if random_downsample_keep > 0:
        print("random downsample keep %f" % random_downsample_keep)
        indi = np.random.random_sample(size=(len(matches))) < random_downsample_keep
        matches = matches[indi, :]

    # plot out the matches
    plot_matches.plot_matches(data.image_as_array(im1),
                              data.image_as_array(im2),
                              p1[matches[:, 0]],
                              p2[matches[:, 1]],
                              True)
    pl.show()

if __name__ == "__main__":
    # some testing code
    im1 = "0500.jpg"
    im2 = "0501.jpg"
    config = {}
    #data = dataset.DataSet("/Volumes/Data/Berkeley/code_and_data/code/egomotion/OpenSfM/data/frames120_minus_blurry")
    data = dataset.DataSet("/Volumes/Data/Berkeley/code_and_data/code/egomotion/OpenSfM/data/eaef92b1-7909-46e1-9f62-5fe6eddcfcac")
    config = {"homography_seg_relative_path":"output/results/joint"}

    #filter_by_segmentation_test(im1, config, data)

    p1, f1, i1 = load_masked(im1, data, config)
    p2, f2, i2 = load_masked(im2, data, config)


    # version with lower ratio test one
    old_ratio = config.get('lowes_ratio', 0.6)
    config["lowes_ratio"] = config.get('homography_lowes_ratio', 0.9)
    matches = matching.match_symmetric(f1, i1, f2, i2, config)
    config["lowes_ratio"] = old_ratio


    # version with more neighbour included
    #matches = homography_match(f1, i1, f2, i2, config)

    homography_match_test(matches, data, p1, p2, .1)
    print(matches.shape)
    p1matched = p1[matches[:, 0], :]
    p2matched = p2[matches[:, 1], :]
    # TODO: the too strict threshold of RANSAC result in fewer correspondence than we thought
    H, inliers = cv2.findHomography(p1matched, p2matched, cv2.RANSAC,
                                    config.get("homography_threshold", 0.004))
    matches = matches[inliers.reshape(-1).astype(np.bool), :]
    print(matches.shape)
    homography_match_test(matches, data, p1, p2, 1)
