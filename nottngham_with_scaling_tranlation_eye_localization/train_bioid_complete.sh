#!/usr/bin/env sh

# examples/BIOID_face/train_bioid.sh examples/BIOID_face/

./build/tools/caffe train --solver=examples/nottingham_new/bioid_solver_finetune.prototxt -weights $1 2>&1 | tee $2
