#!/usr/bin/env bash

rm camera_models.json
rm -r exif
rm profile.log  reconstruction.json  reconstruction.meshed.json  reference_lla.json  tracks.csv
#rm -r root_sift
if [ -e "image_list.txt.with_stop" ]
then
    rm image_list.txt
    mv image_list.txt.with_stop image_list.txt
fi

if [ -d "matches_with_stop" ]
then
    rm -r matches
    mv matches_with_stop matches
fi
