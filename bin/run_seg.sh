#!/usr/bin/env bash

echo $1
echo $2

set -e

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

# rm_generated.sh

if [ ! -f $1"/config.yaml" ];
then
    cp config.yaml $1"/config.yaml"
fi

# mov to images
MOV_PATH=$(ls $1 | grep mov)
opensfm/car/video_to_images.sh $1"/"$MOV_PATH $1"/images"

if [ ! -d $1"/images/output/results/joint" ];
then
    opensfm/car/generate_segmentation.sh $1"/images" 720 $2
fi
