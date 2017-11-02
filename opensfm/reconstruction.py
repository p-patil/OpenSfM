# -*- coding: utf-8 -*-
"""Incremental reconstruction pipeline"""

import datetime
import logging
from itertools import combinations

import numpy as np
import cv2
import pyopengv
import time
from multiprocessing import Pool

from opensfm import align
from opensfm import csfm
from opensfm import geo
from opensfm import matching
from opensfm import multiview
from opensfm import types
from itertools import combinations
import bisect
from sklearn import linear_model
from sklearn.decomposition import PCA
import os
from PIL import Image

logger = logging.getLogger(__name__)


def bundle(graph, reconstruction, gcp, config, fix_cameras=False):
    """Bundle adjust a reconstruction."""
    start = time.time()
    ba = csfm.BundleAdjuster()

    # add all cameras
    for camera in reconstruction.cameras.values():
        if camera.projection_type == 'perspective':
            ba.add_perspective_camera(
                str(camera.id), camera.focal, camera.k1, camera.k2,
                camera.focal_prior, camera.k1_prior, camera.k2_prior,
                fix_cameras)

        elif camera.projection_type in ['equirectangular', 'spherical']:
            ba.add_equirectangular_camera(str(camera.id))

    # add all the shots with their initial values
    for shot in reconstruction.shots.values():
        r = shot.pose.rotation
        t = shot.pose.translation
        ba.add_shot(
            str(shot.id), str(shot.camera.id),
            r[0], r[1], r[2],
            t[0], t[1], t[2],
            False
        )
    # add in all 3D points in the reconstruction now
    for point in reconstruction.points.values():
        x = point.coordinates
        ba.add_point(str(point.id), x[0], x[1], x[2], False)

    # add in all observations
    for shot_id in reconstruction.shots:
        if shot_id in graph:
            for track in graph[shot_id]:
                if track in reconstruction.points:
                    ba.add_observation(str(shot_id), str(track),
                                       *graph[shot_id][track]['feature'])

    if config['bundle_use_gps']:
        for shot in reconstruction.shots.values():
            g = shot.metadata.gps_position
            ba.add_position_prior(str(shot.id), g[0], g[1], g[2],
                                  shot.metadata.gps_dop)

    if config['bundle_use_gcp'] and gcp:
        for observation in gcp:
            if observation.shot_id in reconstruction.shots:
                ba.add_ground_control_point_observation(
                    str(observation.shot_id),
                    observation.coordinates[0],
                    observation.coordinates[1],
                    observation.coordinates[2],
                    observation.shot_coordinates[0],
                    observation.shot_coordinates[1])

    # set loss function, reprojection error, camera internal parameter sd
    ba.set_loss_function(config.get('loss_function', 'SoftLOneLoss'),
                         config.get('loss_function_threshold', 1))
    ba.set_reprojection_error_sd(config.get('reprojection_error_sd', 0.004))
    ba.set_internal_parameters_prior_sd(
        config.get('exif_focal_sd', 0.01),
        config.get('radial_distorsion_k1_sd', 0.01),
        config.get('radial_distorsion_k2_sd', 0.01))

    setup = time.time()

    ba.set_num_threads(config['processes'])
    ba.run()

    run = time.time()
    logger.debug(ba.brief_report())

    # extract all values from the bundler and assign them to the reconstruction
    for camera in reconstruction.cameras.values():
        if camera.projection_type == 'perspective':
            c = ba.get_perspective_camera(str(camera.id))
            camera.focal = c.focal
            camera.k1 = c.k1
            camera.k2 = c.k2

    for shot in reconstruction.shots.values():
        s = ba.get_shot(str(shot.id))
        shot.pose.rotation = [s.rx, s.ry, s.rz]
        shot.pose.translation = [s.tx, s.ty, s.tz]

    for point in reconstruction.points.values():
        p = ba.get_point(str(point.id))
        point.coordinates = [p.x, p.y, p.z]
        point.reprojection_error = p.reprojection_error

    teardown = time.time()

    logger.debug('Bundle setup/run/teardown {0}/{1}/{2}'.format(
        setup - start, run - setup, teardown - run))


def bundle_single_view(graph, reconstruction, shot_id, config):
    """Bundle adjust a single camera."""
    ba = csfm.BundleAdjuster()
    shot = reconstruction.shots[shot_id]
    camera = shot.camera

    if camera.projection_type == 'perspective':
        ba.add_perspective_camera(
            str(camera.id), camera.focal, camera.k1, camera.k2,
            camera.focal_prior, camera.k1_prior, camera.k2_prior, True)
    elif camera.projection_type in ['equirectangular', 'spherical']:
        ba.add_equirectangular_camera(str(camera.id))

    r = shot.pose.rotation
    t = shot.pose.translation
    ba.add_shot(
        str(shot.id), str(camera.id),
        r[0], r[1], r[2],
        t[0], t[1], t[2],
        False
    )

    for track_id in graph[shot_id]:
        if track_id in reconstruction.points:
            track = reconstruction.points[track_id]
            x = track.coordinates
            ba.add_point(str(track_id), x[0], x[1], x[2], True)
            ba.add_observation(str(shot_id), str(track_id),
                               *graph[shot_id][track_id]['feature'])

    if config['bundle_use_gps']:
        g = shot.metadata.gps_position
        ba.add_position_prior(shot.id, g[0], g[1], g[2],
                              shot.metadata.gps_dop)

    ba.set_loss_function(config.get('loss_function', 'SoftLOneLoss'),
                         config.get('loss_function_threshold', 1))
    ba.set_reprojection_error_sd(config.get('reprojection_error_sd', 0.004))
    ba.set_internal_parameters_prior_sd(
        config.get('exif_focal_sd', 0.01),
        config.get('radial_distorsion_k1_sd', 0.01),
        config.get('radial_distorsion_k2_sd', 0.01))

    ba.set_num_threads(config['processes'])
    ba.run()

    s = ba.get_shot(str(shot_id))
    shot.pose.rotation = [s.rx, s.ry, s.rz]
    shot.pose.translation = [s.tx, s.ty, s.tz]


def bundle_local(graph, reconstruction, config, n_neighbour, gcp, shot_id):
    """Bundle adjust a reconstruction."""
    start = time.time()
    ba = csfm.BundleAdjuster()

    # figure out which cameras to add
    # each value in reconstruction.cameras.values() is a camera
    all_images = sorted([shot.id for shot in reconstruction.shots.values()])
    loc = bisect.bisect(all_images, shot_id)
    # get the local images
    local_ids = all_images[max(0, loc - n_neighbour): min(len(all_images), loc + n_neighbour)]
    print("using bundle local, with shot=%s, and local being:" % shot_id)
    print(local_ids)
    local_ids = set(local_ids)

    # add all cameras
    for camera in reconstruction.cameras.values():
        in_recon = camera.id in local_ids

        if camera.projection_type == 'perspective':
            ba.add_perspective_camera(
                str(camera.id), camera.focal, camera.k1, camera.k2,
                camera.focal_prior, camera.k1_prior, camera.k2_prior,
                not in_recon)

        elif camera.projection_type in ['equirectangular', 'spherical']:
            ba.add_equirectangular_camera(str(camera.id))

    # add all the shots with their initial values
    for shot in reconstruction.shots.values():
        in_recon = shot.id in local_ids

        r = shot.pose.rotation
        t = shot.pose.translation
        ba.add_shot(
            str(shot.id), str(shot.camera.id),
            r[0], r[1], r[2],
            t[0], t[1], t[2],
            not in_recon
        )

    var_points = set()
    for shot_id in local_ids:
        for track_id in graph[shot_id]:
            if track_id in reconstruction.points:
                if track_id not in var_points:
                    var_points.add(track_id)
                track = reconstruction.points[track_id]
                x = track.coordinates
                ba.add_point(str(track_id), x[0], x[1], x[2], False)
                ba.add_observation(str(shot_id), str(track_id),
                                   *graph[shot_id][track_id]['feature'])

    for shot_id in reconstruction.shots:
        if (shot_id in graph) and (shot_id not in local_ids):
            for track in graph[shot_id]:
                if track in var_points:
                    ba.add_observation(str(shot_id), str(track),
                                       *graph[shot_id][track]['feature'])

    if config['bundle_use_gps']:
        for shot in reconstruction.shots.values():
            g = shot.metadata.gps_position
            ba.add_position_prior(str(shot.id), g[0], g[1], g[2],
                                  shot.metadata.gps_dop)

    if config['bundle_use_gcp'] and gcp:
        for observation in gcp:
            if observation.shot_id in reconstruction.shots:
                ba.add_ground_control_point_observation(
                    str(observation.shot_id),
                    observation.coordinates[0],
                    observation.coordinates[1],
                    observation.coordinates[2],
                    observation.shot_coordinates[0],
                    observation.shot_coordinates[1])

    # set loss function, reprojection error, camera internal parameter sd
    ba.set_loss_function(config.get('loss_function', 'SoftLOneLoss'),
                         config.get('loss_function_threshold', 1))
    ba.set_reprojection_error_sd(config.get('reprojection_error_sd', 0.004))
    ba.set_internal_parameters_prior_sd(
        config.get('exif_focal_sd', 0.01),
        config.get('radial_distorsion_k1_sd', 0.01),
        config.get('radial_distorsion_k2_sd', 0.01))

    setup = time.time()

    ba.set_num_threads(config['processes'])
    ba.run()

    run = time.time()
    logger.debug(ba.brief_report())

    # extract all values from the bundler and assign them to the reconstruction
    for camera in reconstruction.cameras.values():
        if camera.id in local_ids:
            if camera.projection_type == 'perspective':
                c = ba.get_perspective_camera(str(camera.id))
                camera.focal = c.focal
                camera.k1 = c.k1
                camera.k2 = c.k2

    for shot in reconstruction.shots.values():
        if shot.id in local_ids:
            s = ba.get_shot(str(shot.id))
            shot.pose.rotation = [s.rx, s.ry, s.rz]
            shot.pose.translation = [s.tx, s.ty, s.tz]

    for point in reconstruction.points.values():
        if point.id in var_points:
            p = ba.get_point(str(point.id))
            point.coordinates = [p.x, p.y, p.z]
            point.reprojection_error = p.reprojection_error

    teardown = time.time()

    logger.debug('Bundle setup/run/teardown {0}/{1}/{2}'.format(
        setup - start, run - setup, teardown - run))


def pairwise_reconstructability(common_tracks, homography_inliers):
    """Likeliness of an image pair giving a good initial reconstruction."""
    # pairwise reconstructability probably has to care about the change of angles between the two images
    outliers = common_tracks - homography_inliers
    outlier_ratio = float(outliers) / common_tracks
    # Yang: I think this less comparison is correct, original seems revert
    if outlier_ratio > 0.3:
        return common_tracks
    else:
        return 0


def compute_image_pairs(track_dict, config):
    """All matched image pairs sorted by reconstructability."""
    args = _pair_reconstructability_arguments(track_dict, config)
    processes = config.get('processes', 1)
    if processes == 1:
        result = map(_compute_pair_reconstructability, args)
    else:
        p = Pool(processes)
        result = p.map(_compute_pair_reconstructability, args)
    pairs = [(im1, im2) for im1, im2, r in result if r > 0]
    score = [r for im1, im2, r in result if r > 0]
    order = np.argsort(-np.array(score))
    return [pairs[o] for o in order]


def _pair_reconstructability_arguments(track_dict, config):
    threshold = config.get('homography_threshold', 0.004)
    return [(threshold, im1, im2, p1, p2)
            for (im1, im2), (tracks, p1, p2) in track_dict.items()]


def _compute_pair_reconstructability(args):
    # TODO: why reconstructability is computed using findHomography?
    threshold, im1, im2, p1, p2 = args
    H, inliers = cv2.findHomography(p1, p2, cv2.RANSAC, threshold)
    r = pairwise_reconstructability(len(p1), inliers.sum())
    return (im1, im2, r)


def get_image_metadata(data, image):
    """Get image metadata as a ShotMetadata object."""
    metadata = types.ShotMetadata()
    exif = data.load_exif(image)
    reflla = data.load_reference_lla()
    if ('gps' in exif and
            'latitude' in exif['gps'] and
            'longitude' in exif['gps']):
        lat = exif['gps']['latitude']
        lon = exif['gps']['longitude']
        if data.config.get('use_altitude_tag', False):
            alt = exif['gps'].get('altitude', 2.0)
        else:
            alt = 2.0  # Arbitrary value used to align the reconstruction
        x, y, z = geo.topocentric_from_lla(
            lat, lon, alt,
            reflla['latitude'], reflla['longitude'], reflla['altitude'])
        metadata.gps_position = [x, y, z]
        metadata.gps_dop = exif['gps'].get('dop', 15.0)
    else:
        metadata.gps_position = [0.0, 0.0, 0.0]
        metadata.gps_dop = 999999.0

    metadata.orientation = exif.get('orientation', 1)

    if 'accelerometer' in exif:
        metadata.accelerometer = exif['accelerometer']

    if 'compass' in exif:
        metadata.compass = exif['compass']

    if 'capture_time' in exif:
        metadata.capture_time = exif['capture_time']

    if 'skey' in exif:
        metadata.skey = exif['skey']

    return metadata


def _two_view_reconstruction_inliers(b1, b2, R, t, threshold):
    p = pyopengv.triangulation_triangulate(b1, b2, t, R)

    br1 = p.copy()
    br1 /= np.linalg.norm(br1, axis=1)[:, np.newaxis]

    br2 = R.T.dot((p - t).T).T
    br2 /= np.linalg.norm(br2, axis=1)[:, np.newaxis]

    ok1 = np.linalg.norm(br1 - b1, axis=1) < threshold
    ok2 = np.linalg.norm(br2 - b2, axis=1) < threshold
    return np.nonzero(ok1 * ok2)[0]


def run_relative_pose_ransac(b1, b2, method, threshold, iterations):
    return pyopengv.relative_pose_ransac(b1, b2, method, threshold, iterations)


def run_relative_pose_optimize_nonlinear(b1, b2, t, R):
    return pyopengv.relative_pose_optimize_nonlinear(b1, b2, t, R)


def two_view_reconstruction(p1, p2, camera1, camera2, threshold):
    """Reconstruct two views from point correspondences.

    Args:
        p1, p2: lists points in the images
        camera1, camera2: Camera models
        threshold: reprojection error threshold

    Returns:
        rotation, translation and inlier list
    """
    # assume we know the camera models, convert the 2D coordinates to 3-dim homogeneous coordinates
    b1 = camera1.pixel_bearings(p1)
    b2 = camera2.pixel_bearings(p2)

    # TODO: read opengv docs later
    # Note on threshold:
    # See opengv doc on thresholds here:
    #   http://laurentkneip.github.io/opengv/page_how_to_use.html
    # Here we arbitrarily assume that the threshold is given for a camera of
    # focal length 1.  Also, arctan(threshold) \approx threshold since
    # threshold is small

    # 1. get relative pose using RANSAC
    # TODO: TUNE THIS THRESHOLD
    T = run_relative_pose_ransac(
        b1, b2, "STEWENIUS", 1 - np.cos(threshold), 1000)
    R = T[:, :3]
    t = T[:, 3]
    # 2. get which keypoint pairs are within the threshold
    inliers = _two_view_reconstruction_inliers(b1, b2, R, t, threshold)

    # 3. run the non linear optimizations on the inliers to get more accurate estimations
    T = run_relative_pose_optimize_nonlinear(b1[inliers], b2[inliers], t, R)
    R = T[:, :3]
    t = T[:, 3]
    # 4. refine a second time after the non linear optimizations
    inliers = _two_view_reconstruction_inliers(b1, b2, R, t, threshold)

    return cv2.Rodrigues(R.T)[0].ravel(), -R.T.dot(t), inliers


# this is only used by another not used function
def _two_view_rotation_inliers(b1, b2, R, threshold):
    br2 = R.dot(b2.T).T
    ok = np.linalg.norm(br2 - b1, axis=1) < threshold
    return np.nonzero(ok)[0]

# this is not used
def two_view_reconstruction_rotation_only(p1, p2, camera1, camera2, threshold):
    """Find rotation between two views from point correspondences.

    Args:
        p1, p2: lists points in the images
        camera1, camera2: Camera models
        threshold: reprojection error threshold

    Returns:
        rotation and inlier list
    """
    b1 = camera1.pixel_bearings(p1)
    b2 = camera2.pixel_bearings(p2)

    R = pyopengv.relative_pose_ransac_rotation_only(
        b1, b2, 1 - np.cos(threshold), 1000)
    inliers = _two_view_rotation_inliers(b1, b2, R, threshold)

    return cv2.Rodrigues(R.T)[0].ravel(), inliers


def bootstrap_reconstruction(data, graph, im1, im2, p1, p2):
    """Start a reconstruction using two shots."""
    logger.info("Starting reconstruction with {} and {}".format(im1, im2))
    d1 = data.load_exif(im1)
    d2 = data.load_exif(im2)
    cameras = data.load_camera_models()
    camera1 = cameras[d1['camera']]
    camera2 = cameras[d2['camera']]

    logger.info("Common tracks: {}".format(len(p1)))

    thresh = data.config.get('five_point_algo_threshold', 0.006)
    min_inliers = data.config.get('five_point_algo_min_inliers', 50)
    # TODO: why we are using five point methods? there is still possibility that
    # TODO: input calibration from visualSFM
    # the camera info is not complete, the estimated camera models are 0.0 default.
    R, t, inliers = two_view_reconstruction(p1, p2, camera1, camera2, thresh)
    if len(inliers) > 5:
        logger.info("Two-view reconstruction inliers {}".format(len(inliers)))
        reconstruction = types.Reconstruction()
        reconstruction.cameras = cameras

        shot1 = types.Shot()
        shot1.id = im1
        shot1.camera = cameras[str(d1['camera'])]
        shot1.pose = types.Pose()
        shot1.metadata = get_image_metadata(data, im1)
        reconstruction.add_shot(shot1)

        shot2 = types.Shot()
        shot2.id = im2
        shot2.camera = cameras[str(d2['camera'])]
        shot2.pose = types.Pose(R, t)
        shot2.metadata = get_image_metadata(data, im2)
        reconstruction.add_shot(shot2)

        # triangulate the remaining keypoints (that is not included in two_view_reconstruction) in im1
        triangulate_shot_features(
            graph, reconstruction, im1,
            data.config.get('triangulation_threshold', 0.004),
            data.config.get('triangulation_min_ray_angle', 2.0))
        logger.info("Triangulated: {}".format(len(reconstruction.points)))
        if len(reconstruction.points) > min_inliers:
            # only bundle the second image
            bundle_single_view(graph, reconstruction, im2, data.config)
            # retriangulate all points in all images
            retriangulate(graph, reconstruction, data.config)
            # refine the second image again
            bundle_single_view(graph, reconstruction, im2, data.config)
            return reconstruction

    logger.info("Starting reconstruction with {} and {} failed")


def reconstructed_points_for_images(graph, reconstruction, images):
    """Number of reconstructed points visible on each image, that is not in the reconstruction.

    Returns:
        A list of (image, num_point) pairs sorted by decreasing number
        of points.
    """
    res = []
    for image in images:
        if image not in reconstruction.shots:
            common_tracks = 0
            for track in graph[image]:
                if track in reconstruction.points:
                    common_tracks += 1
            res.append((image, common_tracks))
    return sorted(res, key=lambda x: -x[1])


def resect(data, graph, reconstruction, shot_id):
    """Try resecting and adding a shot to the reconstruction.

    Return:
        True on success.
    """
    exif = data.load_exif(shot_id)
    camera = reconstruction.cameras[exif['camera']]

    # 1. collect all tracks that is in the reconstruction and this image
    # pixel bearing and reconstructed 3D positions
    bs = []
    Xs = []
    for track in graph[shot_id]:
        if track in reconstruction.points:
            x = graph[track][shot_id]['feature']
            b = camera.pixel_bearing(x)
            bs.append(b)
            Xs.append(reconstruction.points[track].coordinates)
    bs = np.array(bs)
    Xs = np.array(Xs)
    if len(bs) < 5:
        return False

    # 2. estimate the pose of this camera using KNEIP method
    threshold = data.config.get('resection_threshold', 0.004)
    T = pyopengv.absolute_pose_ransac(
        bs, Xs, "KNEIP", 1 - np.cos(threshold), 1000)

    R = T[:, :3]
    t = T[:, 3]

    # 3. reproject all points and figure out which is inliers
    reprojected_bs = R.T.dot((Xs - t).T).T
    reprojected_bs /= np.linalg.norm(reprojected_bs, axis=1)[:, np.newaxis]

    inliers = np.linalg.norm(reprojected_bs - bs, axis=1) < threshold
    ninliers = sum(inliers)

    # 4. output resecting inliners
    logger.info("{} resection inliers: {} / {}".format(
        shot_id, ninliers, len(bs)))
    if ninliers >= data.config.get('resection_min_inliers', 15):
        # 5. if inliers are enough, then add this shot to the reconstruction
        R = T[:, :3].T
        t = -R.dot(T[:, 3])
        shot = types.Shot()
        shot.id = shot_id
        shot.camera = camera
        shot.pose = types.Pose()
        shot.pose.set_rotation_matrix(R)
        shot.pose.translation = t
        shot.metadata = get_image_metadata(data, shot_id)
        reconstruction.add_shot(shot)

        # 6. and do single view bundle adjustment
        bundle_single_view(graph, reconstruction, shot_id, data.config)
        return True
    else:
        return False


class TrackTriangulator:
    """Triangulate tracks in a reconstruction.

    Caches shot origin and rotation matrix
    """

    def __init__(self, graph, reconstruction):
        """Build a triangulator for a specific reconstruction."""
        self.graph = graph
        self.reconstruction = reconstruction
        self.origins = {}
        self.rotation_inverses = {}
        self.Rts = {}

    def triangulate(self, track, reproj_threshold, min_ray_angle_degrees, return_reason=False):
        """Triangulate a track and add point to reconstruction."""
        os, bs = [], []
        for shot_id in self.graph[track]:
            # This will not add in new image, it will only triangulate the shots that are included
            if shot_id in self.reconstruction.shots:
                # The formed track
                # one track, and the subset of the images in reconstruction right now
                shot = self.reconstruction.shots[shot_id]
                os.append(self._shot_origin(shot))
                x = self.graph[track][shot_id]['feature']
                b = shot.camera.pixel_bearing(np.array(x))
                r = self._shot_rotation_inverse(shot)
                bs.append(r.dot(b))

        if len(os) >= 2:
            # error and triangulated 3D point
            e, X = csfm.triangulate_bearings_midpoint(
                os, bs, reproj_threshold, np.radians(min_ray_angle_degrees))
            if X is not None:
                point = types.Point()
                point.id = track
                point.coordinates = X.tolist()
                self.reconstruction.add_point(point)
        else:
            e = 4

        if return_reason:
            return e
        '''
        TRIANGULATION_OK = 0,
        TRIANGULATION_SMALL_ANGLE = 1,
        TRIANGULATION_BEHIND_CAMERA = 2, # this is never used
        TRIANGULATION_BAD_REPROJECTION = 3
        the track don't have enough points (<=1) in this reconstruction = 4
        '''

    # this is not used
    def triangulate_dlt(self, track, reproj_threshold, min_ray_angle_degrees):
        """Triangulate track using DLT and add point to reconstruction."""
        Rts, bs = [], []
        for shot_id in self.graph[track]:
            if shot_id in self.reconstruction.shots:
                shot = self.reconstruction.shots[shot_id]
                Rts.append(self._shot_Rt(shot))
                x = self.graph[track][shot_id]['feature']
                b = shot.camera.pixel_bearing(np.array(x))
                bs.append(b)

        if len(Rts) >= 2:
            e, X = csfm.triangulate_bearings_dlt(
                Rts, bs, reproj_threshold, np.radians(min_ray_angle_degrees))
            if X is not None:
                point = types.Point()
                point.id = track
                point.coordinates = X.tolist()
                self.reconstruction.add_point(point)

    def _shot_origin(self, shot):
        if shot.id in self.origins:
            return self.origins[shot.id]
        else:
            o = shot.pose.get_origin()
            self.origins[shot.id] = o
            return o

    def _shot_rotation_inverse(self, shot):
        if shot.id in self.rotation_inverses:
            return self.rotation_inverses[shot.id]
        else:
            r = shot.pose.get_rotation_matrix().T
            self.rotation_inverses[shot.id] = r
            return r

    def _shot_Rt(self, shot):
        if shot.id in self.Rts:
            return self.Rts[shot.id]
        else:
            r = shot.pose.get_Rt()
            self.Rts[shot.id] = r
            return r


def triangulate_shot_features(graph, reconstruction, shot_id, reproj_threshold,
                              min_ray_angle):
    """Reconstruct as many tracks seen in shot_id as possible."""
    triangulator = TrackTriangulator(graph, reconstruction)

    for track in graph[shot_id]:
        # only consider the tracks that is not in the current reconstruction
        if track not in reconstruction.points:
            triangulator.triangulate(track, reproj_threshold, min_ray_angle)


def retriangulate(graph, reconstruction, config):
    """Retrianguate all points in all images"""
    threshold = config.get('triangulation_threshold', 0.004)
    min_ray_angle = config.get('triangulation_min_ray_angle', 2.0)
    triangulator = TrackTriangulator(graph, reconstruction)
    tracks, images = matching.tracks_and_images(graph)
    for track in tracks:
        triangulator.triangulate(track, threshold, min_ray_angle)


def remove_outliers(graph, reconstruction, config):
    """Remove 3D points in reconstruction with large reprojection error."""
    threshold = config.get('bundle_outlier_threshold', 0.008)
    if threshold > 0:
        outliers = []
        for track in reconstruction.points:
            error = reconstruction.points[track].reprojection_error
            if error > threshold:
                outliers.append(track)
        for track in outliers:
            del reconstruction.points[track]
        logger.info("Removed outliers: {}".format(len(outliers)))


def shot_lla_and_compass(shot, reference):
    """Lat, lon, alt and compass of the reconstructed shot position."""
    topo = shot.pose.get_origin()
    lat, lon, alt = geo.lla_from_topocentric(
        topo[0], topo[1], topo[2],
        reference['latitude'], reference['longitude'], reference['altitude'])

    dz = shot.viewing_direction()
    angle = np.rad2deg(np.arctan2(dz[0], dz[1]))
    angle = (angle + 360) % 360
    return lat, lon, alt, angle


def merge_two_reconstructions(r1, r2, config, threshold=1):
    """Merge two reconstructions with common tracks."""
    t1, t2 = r1.points, r2.points
    common_tracks = list(set(t1) & set(t2))

    if len(common_tracks) > 6:

        # Estimate similarity transform
        p1 = np.array([t1[t].coordinates for t in common_tracks])
        p2 = np.array([t2[t].coordinates for t in common_tracks])

        T, inliers = multiview.fit_similarity_transform(
            p1, p2, max_iterations=1000, threshold=threshold)

        if len(inliers) >= 10:
            s, A, b = multiview.decompose_similarity_transform(T)
            r1p = r1
            align.apply_similarity(r1p, s, A, b)
            r = r2
            r.shots.update(r1p.shots)
            r.points.update(r1p.points)
            align.align_reconstruction(r, None, config)
            return [r]
        else:
            return [r1, r2]
    else:
        return [r1, r2]


def merge_reconstructions(reconstructions, config):
    """Greedily merge reconstructions with common tracks."""
    num_reconstruction = len(reconstructions)
    ids_reconstructions = np.arange(num_reconstruction)
    remaining_reconstruction = ids_reconstructions
    reconstructions_merged = []
    num_merge = 0

    for (i, j) in combinations(ids_reconstructions, 2):
        if (i in remaining_reconstruction) and (j in remaining_reconstruction):
            r = merge_two_reconstructions(
                reconstructions[i], reconstructions[j], config)
            if len(r) == 1:
                remaining_reconstruction = list(set(
                    remaining_reconstruction) - set([i, j]))
                for k in remaining_reconstruction:
                    rr = merge_two_reconstructions(r[0], reconstructions[k],
                                                   config)
                    if len(r) == 2:
                        break
                    else:
                        r = rr
                        remaining_reconstruction = list(set(
                            remaining_reconstruction) - set([k]))
                reconstructions_merged.append(r[0])
                num_merge += 1

    for k in remaining_reconstruction:
        reconstructions_merged.append(reconstructions[k])

    logger.info("Merged {0} reconstructions".format(num_merge))

    return reconstructions_merged


def paint_reconstruction(data, graph, reconstruction):
    """Set the color of the points from the color of the tracks."""
    for k, point in reconstruction.points.iteritems():
        point.color = graph[k].values()[0]['feature_color']


class ShouldBundle:
    """Helper to keep track of when to run bundle."""

    def __init__(self, data, reconstruction):
        self.interval = data.config.get('bundle_interval', 0)
        self.new_points_ratio = data.config.get('bundle_new_points_ratio', 1.2)
        self.done(reconstruction)

    def should(self, reconstruction):
        # should condition:
        # either: number of shots has growed more than "bundle_interval"
        # or    : number of points has grow to "bundle_new_points_ratio" * original points
        max_points = self.num_points_last * self.new_points_ratio
        max_shots = self.num_shots_last + self.interval
        return (len(reconstruction.points) >= max_points or
                len(reconstruction.shots) >= max_shots)

    def done(self, reconstruction):
        self.num_points_last = len(reconstruction.points)
        self.num_shots_last = len(reconstruction.shots)


class ShouldRetriangulate:
    """Helper to keep track of when to re-triangulate."""

    def __init__(self, data, reconstruction):
        self.active = data.config.get('retriangulation', False)
        self.ratio = data.config.get('retriangulation_ratio', 1.25)
        self.done(reconstruction)

    def should(self, reconstruction):
        # should retriangulate condition:
        # both: "retriangulation"=true
        # and : the number of 3D points in reconstruction has growed more than "retriangulation_ratio"*original
        max_points = self.num_points_last * self.ratio
        return self.active and len(reconstruction.points) > max_points

    def done(self, reconstruction):
        self.num_points_last = len(reconstruction.points)

def grow_reconstruction(data, graph, reconstruction, images, gcp):
    """Incrementally add shots to an initial reconstruction."""
    bundle(graph, reconstruction, None, data.config)
    # align the reconstruction points to the ground controlling points
    align.align_reconstruction(reconstruction, gcp, data.config)

    should_bundle = ShouldBundle(data, reconstruction)
    should_retriangulate = ShouldRetriangulate(data, reconstruction)
    bundle_local_neighbour = data.config.get("bundle_local_neighbour", 0)

    while True:
        if data.config.get('save_partial_reconstructions', False):
            paint_reconstruction(data, graph, reconstruction)
            data.save_reconstruction(
                [reconstruction], 'reconstruction.{}.json'.format(
                    datetime.datetime.now().isoformat().replace(':', '_')))

        # A list of (image, num_point) pairs sorted by decreasing number of points.
        common_tracks = reconstructed_points_for_images(graph, reconstruction,
                                                        images)
        if not common_tracks:
            break

        logger.info("-------------------------------------------------------")
        # go through the order of overlapping most
        for image, num_tracks in common_tracks:
            if resect(data, graph, reconstruction, image):
                logger.info("Adding {0} to the reconstruction".format(image))
                images.remove(image)

                # TODO: looser threshold is better, but we haven't determined which is best
                triangulate_shot_features(
                    graph, reconstruction, image,
                    0.032,
                    data.config.get('triangulation_min_ray_angle', 2.0))
                '''
                triangulate_shot_features(
                    graph, reconstruction, image,
                    data.config.get('triangulation_threshold', 0.004),
                    data.config.get('triangulation_min_ray_angle', 2.0))
                '''
                if should_bundle.should(reconstruction):
                    if bundle_local_neighbour > 0:
                        bundle_local(graph, reconstruction, data.config,
                                     bundle_local_neighbour, None, image)
                    else:
                        bundle(graph, reconstruction, None, data.config)
                    # delete the keypoints with large reprojection errors.
                    # should not be a big problem, since not removing a lot in output
                    remove_outliers(graph, reconstruction, data.config)
                    align.align_reconstruction(reconstruction, gcp,
                                               data.config)
                    should_bundle.done(reconstruction)

                # the default behavior is not retriangulate
                # TODO: check whether this is needed
                if should_retriangulate.should(reconstruction):
                    logger.info("Re-triangulating")
                    retriangulate(graph, reconstruction, data.config)
                    if bundle_local_neighbour > 0:
                        bundle_local(graph, reconstruction, data.config,
                                     bundle_local_neighbour, None, image)
                    else:
                        bundle(graph, reconstruction, None, data.config)
                    should_retriangulate.done(reconstruction)
                break
        else:
            logger.info("Some images can not be added")
            break

    logger.info("-------------------------------------------------------")

    bundle(graph, reconstruction, gcp, data.config)
    align.align_reconstruction(reconstruction, gcp, data.config)
    paint_reconstruction(data, graph, reconstruction)
    return reconstruction


def incremental_reconstruction(data):
    """Run the entire incremental reconstruction pipeline."""
    logger.info("Starting incremental reconstruction")

    # load the exif information from the images and convert to internal format
    data.invent_reference_lla()

    # return an nx graph, with two kind of nodes, images, and tracks. features are keypoint locations
    graph = data.load_tracks_graph()

    # all tracks and images stored in two lists
    tracks, images = matching.tracks_and_images(graph)
    remaining_images = set(images)
    gcp = None

    # otherwise explictly written a ground control point, no such file exists.
    if data.ground_control_points_exist():
        gcp = data.load_ground_control_points()

    # returns a [im1, im2] -> (tracks, im1_features, im2_features)
    common_tracks = matching.all_common_tracks(graph, tracks)
    reconstructions = []

    # return a list of image pairs that sorted by decreasing favorability
    pairs = compute_image_pairs(common_tracks, data.config)
    if len(pairs)==0:
        print("no image pairs available, use all combinations instead")
        pairs = combinations(sorted(remaining_images), 2)
    for im1, im2 in pairs:
        # each time choose two images that both are not in the collection
        # after adding them into the reconstruction, removing them from the set
        # if this if is entered multiple times, then it indicates that multiple
        # reconstructions are found, which is not good.
        if im1 in remaining_images and im2 in remaining_images:
            tracks, p1, p2 = common_tracks[im1, im2]
            # TODO: we have to carefully select which image pairs to use
            # This is only called once
            reconstruction = bootstrap_reconstruction(data, graph, im1, im2, p1, p2)
            if reconstruction:
                remaining_images.remove(im1)
                remaining_images.remove(im2)
                # The main growing process, it doesn't only add in one image, it add in all.
                reconstruction = grow_reconstruction(
                    data, graph, reconstruction, remaining_images, gcp)
                reconstructions.append(reconstruction)
                reconstructions = sorted(reconstructions,
                                         key=lambda x: -len(x.shots))
                data.save_reconstruction(reconstructions)

                # Gather segmentation info about images

                #path_seg = data.data_path + "/images/output/results/frontend_vgg/" + os.path.splitext(im1)[0]+'.png'    
                #im1_seg = Image.open(path_seg)
                #im1_seg = np.array(im1_seg)

                #path_seg = data.data_path + "/images/output/results/frontend_vgg/" + os.path.splitext(im2)[0]+'.png'    
                #im2_seg = Image.open(path_seg)
                #im2_seg = np.array(im2_seg)

                #idx_u1 = p1[:,0] + 0.5
                #idx_v1 = p1[:,1] + 0.5
                #im1_seg = im1_seg[idx_u1.astype(np.int),idx_v1.astype(np.int)]

                #idx_u2 = p2[:,0] + 0.5
                #idx_v2 = p2[:,1] + 0.5
                #im2_seg = im2_seg[idx_u2.astype(np.int),idx_v2.astype(np.int)]
            else:
                print("reconstruction for image %s and %s failed" % (im1, im2))

    for k, r in enumerate(reconstructions):
        logger.info("Reconstruction {}: {} images, {} points".format(
            k, len(r.shots), len(r.points)))
    logger.info("{} partial reconstructions in total.".format(
        len(reconstructions)))

def find_road_points(reconstruction):
    for point in reconstruction.points.values():
        return



def local_regression_plane_ransac(points):
    """
    Computes parameters for a local regression plane using RANSAC
    """
    XY = []
    Z = []
    for point in points:
        coords = point.coordinates
        XY.append(coords[:2])
        Z.append(coords[2])

    ransac = linear_model.RANSACRegressor(
                                        PCA(),
                                        residual_threshold=0.1
                                         )
    ransac.fit(XY, Z)

    inlier_mask = ransac.inlier_mask_
    coeff = model_ransac.estimator_.coef_
    intercept = model_ransac.estimator_.intercept_
    return ransac

def project_3Dpoints_onto_plane(ransac, points):
    """
    Given a ransac regressor for a plane and reconstruction points, project all
    3D points onto the ransac plane
    """
    projected = []
    for point in points:
        coords = point.coordinates
        XY = coords[:2]
        Z = ransac.predict(XY)
        projected.append([XY[0], XY[1], Z])
    return np.asarray(projected)

def fit_bounding_box_to_plane(points):
    """
    Given points on a plane, fit a bounding box to these points
    """
    xmin = np.amin(points, axis=0)
    xmax = np.amax(points, axis=0)
    ymin = np.amin(points, axis=1)
    ymax = np.amax(points, axis=1)
    return [xmin, xmax, ymin, ymax]
