#!/usr/bin/env bash

# Dilation lib path
DILATION_BIN="/home/piyush/Academics/Berkeley/deepdrive/dilation"
PRETRAINED_MODEL=${DILATION_BIN}"/pretrained/dilation10_cityscapes.caffemodel"
# some other flags to be set
GPU=0

# input flags
IMAGES_DIR=$1
IMAGE_HEIGHT=$2

# generate the image list
ls  -d -1 ${IMAGES_DIR}/{*,.*}  | grep -i 'PNG\|JPG\|JPEG' > ${IMAGES_DIR}/images.txt

python ${DILATION_BIN}/test.py frontend \
--work_dir ${IMAGES_DIR}/output \
--image_list ${IMAGES_DIR}/images.txt \
--weights  ${PRETRAINED_MODEL} \
--classes 19 \
--input_size $(( IMAGE_HEIGHT+186+186 )) \
--gpu ${GPU}
