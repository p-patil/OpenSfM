#!/usr/bin/env bash

# Dilation lib path on durian9
#DILATION_BIN="/home/hxu/Reconstruction/bundler_sfm/data/dilation"

# Dilation on Kraken
DILATION_BIN="/home/hxu/dilation"

PRETRAINED_MODEL=${DILATION_BIN}"/pretrained/dilation10_cityscapes.caffemodel"


# input flags
IMAGES_DIR=$1
IMAGE_HEIGHT=$2
GPU=$3

# wait if GPU memory full
while true;
do
    MEM=$(nvidia-smi -i $GPU | grep MiB.*Default | sed -r  's/.*W[^0-9]*([0-9]+).*/\1/')
    if [ "$MEM" -lt "1000" ]; then
        echo "Memory sufficient and breaking at"$MEM
        break
    else
        echo "Memory full at "$MEM
        sleep 5
    fi
done

# generate the image list
ls  -d -1 ${IMAGES_DIR}/{*,.*}  | grep -i 'PNG\|JPG\|JPEG' > ${IMAGES_DIR}/images.txt

python ${DILATION_BIN}/test.py joint \
--work_dir ${IMAGES_DIR}/output \
--image_list ${IMAGES_DIR}/images.txt \
--weights  ${PRETRAINED_MODEL} \
--classes 19 \
--input_size $(( IMAGE_HEIGHT+186+186 )) \
--gpu ${GPU}
