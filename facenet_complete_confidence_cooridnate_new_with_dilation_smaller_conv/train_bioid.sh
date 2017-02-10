#!/usr/bin/env sh

# examples/BIOID_face/train_bioid.sh examples/BIOID_face/

./build/tools/caffe train --solver=examples/nottingham_complete_eye_nose_mouth/bioid_solver.prototxt 2>&1 | tee $1
