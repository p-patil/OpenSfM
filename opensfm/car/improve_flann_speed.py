# conclusion: use the brute force method
# even the auto tuned couldn't have a large acceleration.

# explore the flann parameters to accelerate while not decreasing the performance
import numpy as np
from opensfm import dataset
from opensfm import features
import time
import cv2

def test_pair(im1, im2, mydataset, config_std, config_custom):
    # a test case
    data = dataset.DataSet(mydataset)

    # load the features of two images

    p1, f1, c1 = data.load_features(im1)
    p2, f2, c2 = data.load_features(im2)

    print("running ground truth")
    res_gt = ground_truth(f1, f2,  cv2.NORM_L2 if f1.dtype.type is np.float32 else cv2.NORM_HAMMING)
    print(res_gt.shape)
    print("done")

    res1 = run_once(config_std, "using standard timing %f", f1, f2)
    print("equal ratio of standard is %f, accuracy1 %f" % accuracy(res1, res_gt))
    res2 = run_once(config_custom, "using custom timing %f", f1, f2)
    print("equal ratio of custom   is %f, accuracy1 %f" % accuracy(res2, res_gt))
    print(res1.shape)





def run_once(config, message, f1, f2):
    print(config)
    # check search f2 in f1
    index = build_flann_index(f1, config)
    start = time.time()
    results, dists = index.knnSearch(f2, 2, params={"checks": config["flann_checks"]})
    print(message % (time.time() - start))

    results = np.array(results)
    return results

def build_flann_index(features, config):
    FLANN_INDEX_LINEAR          = 0
    FLANN_INDEX_KDTREE          = 1
    FLANN_INDEX_KMEANS          = 2
    FLANN_INDEX_COMPOSITE       = 3
    FLANN_INDEX_KDTREE_SINGLE   = 4
    FLANN_INDEX_HIERARCHICAL    = 5
    FLANN_INDEX_LSH             = 6
    FLANN_INDEX_SAVED = 254
    FLANN_INDEX_AUTOTUNED = 255

    if features.dtype.type is np.float32:
        FLANN_INDEX_METHOD = FLANN_INDEX_AUTOTUNED
    else:
        FLANN_INDEX_METHOD = FLANN_INDEX_LSH

    '''
    flann_params = dict(algorithm=FLANN_INDEX_METHOD,
                        target_precision=0.9,
                        build_weight=0.01,
                        memory_weight=0,
                        sample_fraction=0.1)

    flann_params = dict(algorithm=FLANN_INDEX_KDTREE,
                        trees=8)

    flann_params = dict(algorithm=FLANN_INDEX_KMEANS,
                        branching=config['flann_branching'],
                        iterations=config['flann_iterations'])

    flann_params = dict(algorithm=FLANN_INDEX_COMPOSITE,
                        branching=config['flann_branching'],
                        iterations=config['flann_iterations'],
                        trees=2)

    flann_params = dict(algorithm=FLANN_INDEX_KDTREE_SINGLE)

    flann_params = dict(algorithm=FLANN_INDEX_HIERARCHICAL,
                        branching=16,
                        trees=8,
                        leaf_size=50)

    flann_params = dict(algorithm=FLANN_INDEX_AUTOTUNED)
    '''
    flann_params = config
    flann_params['algorithm'] = FLANN_INDEX_METHOD

    #flann_params = dict(algorithm=FLANN_INDEX_AUTOTUNED,
    #                    dist = cv2.NORM_HAMMING)

    OPENCV_3 = True
    flann_Index = cv2.flann.Index if OPENCV_3 else cv2.flann_Index
    return flann_Index(features, flann_params)

def ground_truth(f1, f2, distance=cv2.NORM_L2):
    bf = cv2.BFMatcher(distance)

    start = time.time()
    matches = bf.knnMatch(f2, f1, k=2)
    ans = []
    for m, n in matches:
        ans.append([m.trainIdx, n.trainIdx])
    print("time spent for brute force %f" % (time.time() - start))
    ans = np.array(ans)
    return ans


def accuracy(m, gt):
    ind0 = (m == gt)
    ind = np.logical_and(ind0[:, 0], ind0[:, 1])
    equal_ratio = sum(ind) * 1.0 / ind.size
    equal_ratio1 = sum(ind0[:, 0]) * 1.0 / ind0.shape[0]

    return equal_ratio, equal_ratio1

if __name__ == "__main__":
    mydataset = "data_sync/121f"
    im1 = "0128.jpg"
    im2 = "0129.jpg"
    '''
    config_std = {'flann_checks': 200,
                  'flann_iterations': 10,
                  'flann_branching': 16}

    config_custom =  {'flann_checks': 50,
                      'flann_iterations': 10,
                      'flann_branching': 32}
    '''
    config_std = dict(flann_checks= 0,
                        table_number=15,  # 12
                        key_size=20,  # 20
                        multi_probe_level=1)  # 2)
    config_custom = dict(flann_checks= 50,
                        table_number=12,
                      key_size=20,
                      multi_probe_level=2)

    test_pair(im1, im2, mydataset, config_std, config_custom)
