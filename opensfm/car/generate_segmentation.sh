#!/usr/bin/env bash

# Dilation lib path on durian9
DILATION_BIN="/root/bdd/dilation"
# Dilation on Kraken
DILATION_BIN="/root/bdd/dilation"

PRETRAINED_MODEL=${DILATION_BIN}"/pretrained/dilation10_cityscapes.caffemodel"

# input flags
IMAGES_DIR=$1
IMAGE_HEIGHT=$2
GPU_NUM=$3

echo "attempting to generate segmentations"

# wait if GPU memory full
while true;
do
    MEM=$(nvidia-smi -i 0 | grep -P "\d+MiB / \d+MiB" | cut -d "|" -f 3 | cut -d "/" -f 1 | tr -dc "0-9")
    if [ "$MEM" -lt "16000" ]; then
        echo "$MEM GPU memory used, remaining is sufficient"
        break
    else
        echo "GPU memory full at $MEM"
        sleep 5
    fi
done

# generate the image list
ls -d -1 ${IMAGES_DIR}/{*,.*} | grep -i 'PNG\|JPG\|JPEG' > ${IMAGES_DIR}/images.txt

python ${DILATION_BIN}/test.py frontend \
    --work_dir ${IMAGES_DIR}/output \
    --image_list ${IMAGES_DIR}/images.txt \
    --weights  ${PRETRAINED_MODEL} \
    --classes 19 \
    --input_size $(( IMAGE_HEIGHT+186+186 )) \
    --gpu ${GPU_NUM}

echo "exit"
echo
