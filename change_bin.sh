#!/usr/bin/env bash

TARGET=$1

rm opensfm/csfm.so
ln -s opensfm/csfm.so.${TARGET} opensfm/csfm.so
