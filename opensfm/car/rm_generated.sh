#!/usr/bin/env bash

rm camera_models.json
rm -r exif
rm profile.log  reconstruction.json  reconstruction.meshed.json  reference_lla.json  tracks.csv
#rm -r root_sift matches image_list.txt.with_stop
# image_list.txt.with_stop is used as raw list of which part to reconstruct
# this will not enable detect_features to run on a subset
rm image_list.txt
