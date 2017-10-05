#!/usr/bin/env bash
VIDEO_PATH=$1
OUTPUT_PATH=$2

if [ ! -d "$OUTPUT_PATH" ];
then
    echo "video hasn't been converted to images, converting"
    mkdir $OUTPUT_PATH
    ffmpeg -i "$VIDEO_PATH" -qscale:v 3 -threads 8 "$OUTPUT_PATH/%04d.jpg"
fi

echo "exit"
