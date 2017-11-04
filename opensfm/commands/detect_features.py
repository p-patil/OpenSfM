import logging
from multiprocessing import Pool, current_process
import time
import numpy as np
import os
from opensfm import dataset
from opensfm import features

logger = logging.getLogger(__name__)

class Command:
    name = 'detect_features'
    help = 'Compute features for all images'

    def add_arguments(self, parser):
        parser.add_argument('dataset', help='dataset to process')

    def run(self, args):
        print "detecting features"

        data = dataset.DataSet(args.dataset)
        images = data.images()
        arguments = [(image, data) for image in images]

        start = time.time()

        processes = data.config.get('processes', 1)
        processes = 1 # TODO remove
        if processes == 1:
            for arg in arguments:
                t0=time.time()
                detect(arg)
                print("total detect time in %f second" % (time.time()-t0))
        else:
            print "starting pool of %i processes to detect features" % processes
            p = Pool(processes)
            p.map(detect, arguments)

        end = time.time()

        print "done"
        with open(data.profile_log(), 'a') as fout:
            fout.write('detect_features: {0}\n'.format(end - start))
        print "exit\n"

def detect(args):
    id = current_process()._identity
    prefix = "process %s: " % str(id[0]) if len(id) > 0 else ""

    image, data = args
    print prefix + "detecting features for image %s" % image
    logger.info('Extracting {} features for image {}'.format(data.feature_type().upper(), image))
    
    if not data.features_exist(image):
        mask = data.mask_as_array(image)
        if mask is not None:
            print prefix + "found mask for image: %s" % image
            logger.info('Found mask to apply for image {}'.format(image))
    
            # Obtain segmentation path
            path_seg = data.data_path + "/images/output/results/frontend_vgg/" + os.path.splitext(image)[0]+'.png'
        else:
            print prefix + "not found mask for image %s" % image
            path_seg = None

        preemptive_max = data.config.get('preemptive_max', 200)
        the_image = data.image_as_array(image)


        print prefix + "extracting features from image %s" % image
        save_no_mask = False
        all_content = features.extract_features(the_image, data.config, mask,
                                                save_no_mask, path_seg)

        if save_no_mask:
            p_unsorted, f_unsorted, c_unsorted = all_content[0]
            p_nomask, f_nomask, c_nomask = all_content[1]
        else:
            p_unsorted, f_unsorted, c_unsorted = all_content

        if len(p_unsorted) == 0:
            print prefix + "exit"
            return

        # size_nomask = p_nomask[:, 2]
        # order_nomask = np.argsort(size_nomask)
        # p_nomask = p_nomask[order_nomask, :]
        # f_nomask = f_nomask[order_nomask, :]
        # c_nomask = c_nomask[order_nomask, :]
        # p_nomask_pre = p_nomask[-preemptive_max:]
        # f_nomask_pre = f_nomask[-preemptive_max:]
        # data.save_features(image+'_nomask', p_nomask, f_nomask, c_nomask)
        # data.save_preemptive_features(image+'_nomask', p_nomask_pre, f_nomask_pre)
        # index_nomask = features.build_flann_index(f_nomask, data.config)
        # data.save_feature_index(image+'_nomask', index_nomask)

        print prefix + "saving features"
        size = p_unsorted[:, 2]
        order = np.argsort(size)
        p_sorted = p_unsorted[order, :]
        f_sorted = f_unsorted[order, :]
        c_sorted = c_unsorted[order, :]
        data.save_features(image, p_sorted, f_sorted, c_sorted)
        if data.config.get('preemptive_threshold', 0) > 0:
            p_pre = p_sorted[-preemptive_max:]
            f_pre = f_sorted[-preemptive_max:]
            data.save_preemptive_features(image, p_pre, f_pre)

        if data.config.get('matcher_type', "BRUTEFORCE") == "FLANN":
            index = features.build_flann_index(f_sorted, data.config)
            data.save_feature_index(image, index)

        print prefix + "exit"
