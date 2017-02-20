#!/usr/bin/env bash

VIDEO_PATH=$1
OUTPUT_PATH=$2

ffmpeg -i $VIDEO_PATH -qscale:v 3 -threads 8 $OUTPUT_PATH"/%04d.jpg"
