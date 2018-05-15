#!/bin/bash

DATA_PATH="/root/bdd/data/samples-1k/videos"

cd OpenSfM

TOTAL=$(ls "$DATA_PATH" | wc -l)
i=1
for VIDEO in $(ls "$DATA_PATH")
do
    if [[ $(find "$DATA_PATH/$VIDEO" -name "reconstruction.json") ]]
    then
        echo "SKIPPING VIDEO $VIDEO.mov"
	./bin/opensfm undistort "data/samples-1k/videos/$VIDEO" 3
	./bin/opensfm compute_depthmaps "data/samples-1k/videos/$VIDEO" 3
        continue
    fi

    echo "PROCESSING VIDEO $VIDEO.mov - $i of $TOTAL" >> ../processing_samples-1k.log
    ./bin/run_all "data/samples-1k/videos/$VIDEO" 3
    ./bin/opensfm undistort "data/samples-1k/videos/$VIDEO" 3
    ./bin/opensfm compute_depthmaps "data/samples-1k/videos/$VIDEO" 3
    let i++
done

