#!/usr/bin/env bash

TARGET=$1

rm opensfm/csfm.so
ln -s csfm.so.${TARGET} opensfm/csfm.so
