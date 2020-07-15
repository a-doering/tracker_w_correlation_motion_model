#!/bin/bash
echo execution: ./scheduled_training.sh seq1 seq2 ... seqn
echo

BASE_DIR=/content/tracker_w_correlation_motion_model

echo
echo Starting scheduled training...
echo

python -V
nvidia-smi -L
echo

for arg in "$@"
do
    DIR_NAME=val_on_seq_$arg
    echo Validating on sequence $arg...
    echo

    python $BASE_DIR/experiments/scripts/train_correlation.py with correlation.name=$DIR_NAME "correlation.val_seqs=['$arg']"

    mv $BASE_DIR/output/tracktor/correlation/$DIR_NAME  $BASE_DIR/drive_output/scheduled_training/

done

echo
echo Scheduled training finished!
