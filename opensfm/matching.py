import numpy as np
import cv2, logging, pyopengv, time, sys
import networkx as nx
from collections import defaultdict
from itertools import combinations
from opensfm import context
from opensfm.unionfind import UnionFind

logger = logging.getLogger(__name__)

# pairwise matches
def match_lowe(index, f2, config, im1_seg=None, im2_seg=None):
    search_params = dict(checks=config.get('flann_checks', 200))
    # TODO: the saved FLANN index for LSH doesn't work, it will crash here
    results, dists = index.knnSearch(f2, 2, params=search_params)
    print("COMPUTING LOWE")
    for i in range(len(f2)):
        if im1_seg is not None:
            if im1_seg[i] == 0:
                squared_ratio = 0.01**2
            else:
                squared_ratio = config.get('lowes_ratio', 0.6)**2  # Flann returns squared L2 distances
        else:
            squared_ratio = config.get('lowes_ratio', 0.6)**2
    good = dists[:, 0] < squared_ratio * dists[:, 1]
    matches = zip(results[good, 0], good.nonzero()[0])
    return np.array(matches, dtype=int)

def match_symmetric(fi, indexi, fj, indexj, config, im1_seg=None, im2_seg=None):
    t = time.time()
    sys.stdout.write("symmetric matching commencing... ")
    sys.stdout.flush()
    if config.get('matcher_type', 'FLANN') == 'FLANN':
        matches_ij = [(a,b) for a,b in match_lowe(indexi, fj, config, im1_seg,
                                                 im2_seg)]
        matches_ji = [(b,a) for a,b in match_lowe(indexj, fi, config, im2_seg,
                                                 im1_seg)]
    else:
        matches_ij = [(a,b) for a,b in match_lowe_bf(fi, fj, config, im1_seg,
                                                    im2_seg)]

        match1 = np.array(matches_ij)
        fj_new2old = match1[:, 1]
        fj=fj[fj_new2old,:]

        matches_ji = [(b,a) for a,b in match_lowe_bf(fj, fi, config, im2_seg,
                                                     im1_seg)]

        matches_ji = [(a,fj_new2old[b]) for a,b in matches_ji]

    matches = set(matches_ij).intersection(set(matches_ji))
    print("done. matching time: %s seconds" % str(time.time() - t))
    return np.array(list(matches), dtype=int)


def convert_matches_to_vector(matches):
    '''Convert Dmatch object to matrix form
    '''
    matches_vector = np.zeros((len(matches),2),dtype=np.int)
    k = 0
    for mm in matches:
        matches_vector[k,0] = mm.queryIdx
        matches_vector[k,1] = mm.trainIdx
        k = k+1
    return matches_vector

def match_lowe_bf(f1, f2, config, im1_seg=None, im2_seg=None):
    '''Bruteforce feature matching
    '''
    assert(f1.dtype.type==f2.dtype.type)
    if (f1.dtype.type == np.uint8):
        matcher_type = 'BruteForce-Hamming'
    else:
        matcher_type = 'BruteForce'
    matcher = cv2.DescriptorMatcher_create(matcher_type)
    # f1=querys f2=train
    matches = matcher.knnMatch(f1, f2, k=2)
    ratio = config.get('lowes_ratio', 0.6)
    good_matches = []
    counter = 0
    for match in matches:
        if match and len(match) == 2:
            m, n = match
            if im1_seg is not None:
                idx1 = m.queryIdx
                idx2 = n.trainIdx
                if im1_seg[idx1] == 0 and im2_seg[idx2] == 0:
                    ratio = 1
                else:
                    ratio = config.get('lowes_ratio', 0.6)
            else:
                ratio = config.get('lowes_ratio', 0.6)
            if m.distance < ratio * n.distance:
                good_matches.append(m)

    good_matches = convert_matches_to_vector(good_matches)
    return np.array(good_matches, dtype=int)

def robust_match_fundamental(p1, p2, matches, config):
    '''Computes robust matches by estimating the Fundamental matrix via RANSAC.
    '''
    if len(matches) < 8:
        return np.array([])

    p1 = p1[matches[:, 0]][:, :2].copy()
    p2 = p2[matches[:, 1]][:, :2].copy()

    FM_RANSAC = cv2.FM_RANSAC if context.OPENCV3 else cv2.cv.CV_FM_RANSAC
    F, mask = cv2.findFundamentalMat(p1, p2, FM_RANSAC, config.get('robust_matching_threshold', 0.006), 0.9999)
    inliers = mask.ravel().nonzero()

    if F[2,2] == 0.0:
        return []

    return matches[inliers]


def compute_inliers_bearings(b1, b2, T):
    R = T[:, :3]
    t = T[:, 3]
    p = pyopengv.triangulation_triangulate(b1, b2, t, R)

    br1 = p.copy()
    br1 /= np.linalg.norm(br1, axis=1)[:, np.newaxis]

    br2 = R.T.dot((p - t).T).T
    br2 /= np.linalg.norm(br2, axis=1)[:, np.newaxis]

    ok1 = np.linalg.norm(br1 - b1, axis=1) < 0.01   # TODO(pau): compute angular error and use proper threshold
    ok2 = np.linalg.norm(br2 - b2, axis=1) < 0.01
    return ok1 * ok2


def robust_match_calibrated(p1, p2, camera1, camera2, matches, config):
    '''Computes robust matches by estimating the Essential matrix via RANSAC.
    '''

    if len(matches) < 8:
        return np.array([])

    p1 = p1[matches[:, 0]][:, :2].copy()
    p2 = p2[matches[:, 1]][:, :2].copy()
    b1 = camera1.pixel_bearings(p1)
    b2 = camera2.pixel_bearings(p2)

    threshold = config['robust_matching_threshold']
    T = pyopengv.relative_pose_ransac(b1, b2, "STEWENIUS", 1 - np.cos(threshold), 1000)

    inliers = compute_inliers_bearings(b1, b2, T)

    return matches[inliers]


def robust_match(p1, p2, camera1, camera2, matches, config):
    if (camera1.projection_type == 'perspective'
            and camera1.k1 == 0.0
            and camera2.projection_type == 'perspective'
            and camera2.k1 == 0.0):
        return robust_match_fundamental(p1, p2, matches, config)
    else:
        return robust_match_calibrated(p1, p2, camera1, camera2, matches, config)


def good_track(track, min_length):
    if len(track) < min_length:
        return False
    images = [f[0] for f in track]
    # TODO smarter method to filter out inconsistent ways
    if len(images) != len(set(images)):
        return False
    # TODO: add the start end distance constraint
    return True


def create_tracks_graph(features, colors, matches, config):
    logger.debug('Merging features onto tracks')
    uf = UnionFind()
    for im1, im2 in matches:
        for f1, f2 in matches[im1, im2]:
            uf.union((im1, f1), (im2, f2))

    sets = {}
    for i in uf:
        p = uf[i]
        if p in sets:
            sets[p].append(i)
        else:
            sets[p] = [i]

    # one track is a sequence of image 2D keypoints
    tracks = [t for t in sets.values() if good_track(t, config.get('min_track_length', 2))]
    logger.debug('Good tracks: {}'.format(len(tracks)))

    tracks_graph = nx.Graph()
    for track_id, track in enumerate(tracks):
        for image, featureid in track:
            if image not in features:
                continue
            x, y = features[image][featureid]
            r, g, b = colors[image][featureid]
            tracks_graph.add_node(image, bipartite=0)
            tracks_graph.add_node(str(track_id), bipartite=1)
            tracks_graph.add_edge(image,
                                  str(track_id),
                                  feature=(x, y),
                                  feature_id=featureid,
                                  feature_color=(float(r), float(g), float(b)))

    return tracks_graph


def tracks_and_images(graph):
    """List of tracks and images in the graph."""
    tracks, images = [], []
    for n in graph.nodes(data=True):
        if n[1]['bipartite'] == 0:
            images.append(n[0])
        else:
            tracks.append(n[0])
    return tracks, images


def common_tracks(g, im1, im2):
    """
    Return the list of tracks observed in both images
    :param g: Graph structure (networkx) as returned by :method:`DataSet.tracks_graph`
    :param im1: Image name, with extension (i.e. 123.jpg)
    :param im2: Image name, with extension (i.e. 123.jpg)
    :return: tuple: track, feature from first image, feature from second image
    """
    t1, t2 = g[im1], g[im2]
    tracks, p1, p2 = [], [], []
    for track in t1:
        if track in t2:
            p1.append(t1[track]['feature'])
            p2.append(t2[track]['feature'])
            tracks.append(track)
    p1 = np.array(p1)
    p2 = np.array(p2)
    return tracks, p1, p2


def all_common_tracks(graph, tracks, include_features=True, min_common=50):
    """
    Returns a dictionary mapping image pairs to the list of tracks observed in both images
    :param graph: Graph structure (networkx) as returned by :method:`DataSet.tracks_graph`
    :param tracks: list of track identifiers
    :param include_features: whether to include the features from the images
    :param min_common: the minimum number of tracks the two images need to have in common
    :return: tuple: im1, im2 -> tuple: tracks, features from first image, features from second image
    """
    track_dict = defaultdict(list)
    for tr in tracks:
        track_images = sorted(graph[tr].keys())
        for pair in combinations(track_images, 2):
            track_dict[pair].append(tr)
    common_tracks = {}
    for k, v in track_dict.iteritems():
        # k = (image1, image2)
        # v = list of tracks, tracks is kind of obsecure object here.
        if len(v) < min_common:
            continue
        if include_features:
            # graph[image] get the node of that image, and graph[image][track] get the corresponding track
            t1, t2 = graph[k[0]], graph[k[1]]
            p1 = np.array([t1[tr]['feature'] for tr in v])
            p2 = np.array([t2[tr]['feature'] for tr in v])
            common_tracks[k] = (v, p1, p2)
        else:
            common_tracks[k] = v
    return common_tracks
