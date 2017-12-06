import logging
from multiprocessing import Pool
import time
import os
from PIL import Image
import numpy as np

from pathlib2 import Path
from opensfm import dataset
from opensfm import geo
from opensfm import matching
#import opensfm.car.match_homography as homography

logger = logging.getLogger(__name__)


class Command:
    name = 'match_features'
    help = 'Match features between image pairs'

    def add_arguments(self, parser):
        parser.add_argument('dataset', help='dataset to process')

    def run(self, args):
        data = dataset.DataSet(args.dataset)
        # even if there is a matching folder, we have to gone through
        # to make sure we add every new matches into it.
        '''
        if os.path.exists(os.path.join(data.data_path, 'matches')):
            print("found matches folder, skipping")
            return
        '''

        images = data.images()
        exifs = {im: data.load_exif(im) for im in images}
        pairs = match_candidates_from_metadata(images, exifs, data)

        num_pairs = sum(len(c) for c in pairs.values())
        logger.info('Matching {} image pairs'.format(num_pairs))

        ctx = Context()
        ctx.data = data
        ctx.cameras = ctx.data.load_camera_models()
        ctx.exifs = exifs
        ctx.p_pre, ctx.f_pre = load_preemptive_features(data)
        args = match_arguments(pairs, ctx)

        start = time.time()
        processes = ctx.data.config.get('processes', 1)
        print(processes)
        __import__("sys").exit()
        if processes == 1:
            for arg in args:
                match(arg)
        else:
            p = Pool(processes)
            p.map(match, args)
        end = time.time()
        with open(ctx.data.profile_log(), 'a') as fout:
            fout.write('match_features: {0}\n'.format(end - start))


class Context:
    pass


def load_preemptive_features(data):
    p, f = {}, {}
    if data.config['preemptive_threshold'] > 0:
        logger.debug('Loading preemptive data')
        for image in data.images():
            try:
                p[image], f[image] = \
                    data.load_preemtive_features(image)
            except IOError:
                p, f, c = data.load_features(image)
                p[image], f[image] = p, f
            preemptive_max = min(
                data.config.get('preemptive_max',
                                p[image].shape[0]),
                p[image].shape[0])
            p[image] = p[image][:preemptive_max, :]
            f[image] = f[image][:preemptive_max, :]
    return p, f


def has_gps_info(exif):
    return (exif
            and 'gps' in exif
            and 'latitude' in exif['gps']
            and 'longitude' in exif['gps'])


def distance_from_exif(exif1, exif2):
    """Compute distance between images based on exif metadata.

    >>> exif1 = {'gps': {'latitude': 50.0663888889, 'longitude': 5.714722222}}
    >>> exif2 = {'gps': {'latitude': 58.6438888889, 'longitude': 3.070000000}}
    >>> d = distance_from_exif(exif1, exif2)
    >>> abs(d - 968998) < 1
    True
    """
    if has_gps_info(exif1) and has_gps_info(exif2):
        gps1 = exif1['gps']
        gps2 = exif2['gps']
        latlon1 = gps1['latitude'], gps1['longitude']
        latlon2 = gps2['latitude'], gps2['longitude']
        return geo.gps_distance(latlon1, latlon2)
    else:
        return 0


def timediff_from_exif(exif1, exif2):
    return np.fabs(exif1['capture_time'] - exif2['capture_time'])


def match_candidates_from_metadata(images, exifs, data):
    '''
    Compute candidate matching pairs based on GPS and capture time
    '''
    max_distance = data.config['matching_gps_distance']
    max_neighbors = data.config['matching_gps_neighbors']
    max_time_neighbors = data.config['matching_time_neighbors']
    max_order_neighbors = data.config['matching_order_neighbors']

    if not all(map(has_gps_info, exifs.values())) and max_neighbors != 0:
        logger.warn("Not all images have GPS info. "
                    "Disabling matching_gps_neighbors.")
        max_neighbors = 0

    pairs = set()
    images.sort()
    for index1, im1 in enumerate(images):
        distances = []
        timediffs = []
        indexdiffs = []
        for index2, im2 in enumerate(images):
            if im1 != im2:
                dx = distance_from_exif(exifs[im1], exifs[im2])
                dt = timediff_from_exif(exifs[im1], exifs[im2])
                di = abs(index1 - index2)
                if dx <= max_distance:
                    distances.append((dx, im2))
                    timediffs.append((dt, im2))
                    indexdiffs.append((di, im2))
        distances.sort()
        timediffs.sort()
        indexdiffs.sort()

        if max_neighbors or max_time_neighbors or max_order_neighbors:
            distances = distances[:max_neighbors]
            timediffs = timediffs[:max_time_neighbors]
            indexdiffs = indexdiffs[:max_order_neighbors]

        for d, im2 in distances + timediffs + indexdiffs:
            if im1 < im2:
                pairs.add((im1, im2))
            else:
                pairs.add((im2, im1))

    res = {im: [] for im in images}
    for im1, im2 in pairs:
        res[im1].append(im2)
    return res


def match_arguments(pairs, ctx):
    for i, (im, candidates) in enumerate(pairs.items()):
        yield im, candidates, i, len(pairs), ctx

def match(args):
    '''
    Compute all matches for a single image
    '''
    im1, candidates, i, n, ctx = args
    logger.info('Matching {}  -  {} / {}'.format(im1, i + 1, n))

    config = ctx.data.config
    robust_matching_min_match = config['robust_matching_min_match']
    preemptive_threshold = config['preemptive_threshold']
    lowes_ratio = config['lowes_ratio']
    preemptive_lowes_ratio = config['preemptive_lowes_ratio']

    path_seg = ctx.data.data_path + "/images/output/results/frontend_vgg/" + os.path.splitext(im1)[0]+'.png'
    file_name = Path(path_seg)
    if file_name.is_file():
        im1_seg = Image.open(path_seg)
        im1_seg = np.array(im1_seg)
    p1, f1, c1 = ctx.data.load_features(im1)

    # if we are using bruteforce matching, the loaded index will simply be False.
    i1 = ctx.data.load_feature_index(im1, f1)
    if file_name.is_file():
        idx_u1 = im1_seg.shape[1]*(p1[:,0] + 0.5)
        idx_v1 = im1_seg.shape[0]*(p1[:,1] + 0.5)
        im1_seg = im1_seg[idx_v1.astype(np.int),idx_u1.astype(np.int)]
    else:
        im1_seg = None

    if ctx.data.matches_exists(im1):
        im1_matches = ctx.data.load_matches(im1)
    else:
        im1_matches = {}

    for im2 in candidates:
        if im2 in im1_matches:
            continue

        path_seg = ctx.data.data_path + "/images/output/results/frontend_vgg/" + os.path.splitext(im2)[0]+'.png'
        file_name = Path(path_seg)
        if file_name.is_file():
            im2_seg = Image.open(path_seg)
            im2_seg = np.array(im2_seg)

        p2, f2, c2 = ctx.data.load_features(im2)
        i2 = ctx.data.load_feature_index(im2, f2)

        if file_name.is_file():
            idx_u2 = im2_seg.shape[1]*(p2[:,0]+0.5)
            idx_v2 = im2_seg.shape[0]*(p2[:,1]+0.5)
            im2_seg = im2_seg[idx_v2.astype(np.int),idx_u2.astype(np.int)]
        else:
            im2_seg = None

        # preemptive matching
        if preemptive_threshold > 0:
            t = time.time()
            config['lowes_ratio'] = preemptive_lowes_ratio
            matches_pre = matching.match_lowe_bf(
                ctx.f_pre[im1], ctx.f_pre[im2], config, im1_seg, im2_seg)
            config['lowes_ratio'] = lowes_ratio
            logger.debug("Preemptive matching {0}, time: {1}s".format(
                len(matches_pre), time.time() - t))
            if len(matches_pre) < preemptive_threshold:
                logger.debug(
                    "Discarding based of preemptive matches {0} < {1}".format(
                        len(matches_pre), preemptive_threshold))
                continue

        # symmetric matching
        t = time.time()

        matches = matching.match_symmetric(f1, i1, f2, i2, config, im1_seg,
                                           im2_seg)
        logger.debug('{} - {} has {} candidate matches'.format(
            im1, im2, len(matches)))
        if len(matches) < robust_matching_min_match:
            im1_matches[im2] = []
            continue

        # robust matching
        t_robust_matching = time.time()
        camera1 = ctx.cameras[ctx.exifs[im1]['camera']]
        camera2 = ctx.cameras[ctx.exifs[im2]['camera']]

        # add extra matches on the road with homography method
        # filter the candidate points by semantic segmentation

        rmatches = matching.robust_match(p1, p2, camera1, camera2, matches,
                                         config)

        if len(rmatches) < robust_matching_min_match:
            im1_matches[im2] = []
            continue
        im1_matches[im2] = rmatches
        logger.debug('Robust matching time : {0}s'.format(
            time.time() - t_robust_matching))

        logger.debug("Full matching {0} / {1}, time: {2}s".format(
            len(rmatches), len(matches), time.time() - t))
    ctx.data.save_matches(im1, im1_matches)
