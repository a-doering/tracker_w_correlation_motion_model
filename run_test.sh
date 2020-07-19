#!/bin/bash

test_tracktor='python /content/tracker_w_correlation_motion_model/experiments/scripts/test_tracktor.py'

python -V && nvidia-smi -L && $test_tracktor with tracktor.tracker.boxes_enlargement_factor=2.0 tracktor.tracker.do_reid=False tracktor.tracker.do_align=False |& tee drive_output/scheduled_training/2.0_model_tests/noreid_noalign_2.0.txt
python -V && nvidia-smi -L && $test_tracktor with tracktor.tracker.boxes_enlargement_factor=2.0 tracktor.tracker.do_reid=True tracktor.tracker.do_align=False |& tee drive_output/scheduled_training/2.0_model_tests/reid_noalign_2.0.txt
python -V && nvidia-smi -L && $test_tracktor with tracktor.tracker.boxes_enlargement_factor=2.0 tracktor.tracker.do_reid=False tracktor.tracker.do_align=True |& tee drive_output/scheduled_training/2.0_model_tests/noreid_align_2.0.txt
python -V && nvidia-smi -L && $test_tracktor with tracktor.tracker.boxes_enlargement_factor=2.0 tracktor.tracker.do_reid=True tracktor.tracker.do_align=True |& tee drive_output/scheduled_training/2.0_model_tests/reid_align_2.0.txt

python -V && nvidia-smi -L && $test_tracktor with tracktor.tracker.boxes_enlargement_factor=1.5 tracktor.tracker.do_reid=False tracktor.tracker.do_align=False |& tee drive_output/scheduled_training/2.0_model_tests/noreid_noalign_1.5.txt
python -V && nvidia-smi -L && $test_tracktor with tracktor.tracker.boxes_enlargement_factor=1.5 tracktor.tracker.do_reid=True tracktor.tracker.do_align=False |& tee drive_output/scheduled_training/2.0_model_tests/reid_noalign_1.5.txt
python -V && nvidia-smi -L && $test_tracktor with tracktor.tracker.boxes_enlargement_factor=1.5 tracktor.tracker.do_reid=False tracktor.tracker.do_align=True |& tee drive_output/scheduled_training/2.0_model_tests/noreid_align_1.5.txt
python -V && nvidia-smi -L && $test_tracktor with tracktor.tracker.boxes_enlargement_factor=1.5 tracktor.tracker.do_reid=True tracktor.tracker.do_align=True |& tee drive_output/scheduled_training/2.0_model_tests/reid_align_1.5.txt

python -V && nvidia-smi -L && $test_tracktor with tracktor.tracker.boxes_enlargement_factor=1.0 tracktor.tracker.do_reid=False tracktor.tracker.do_align=False |& tee drive_output/scheduled_training/2.0_model_tests/noreid_noalign_1.0.txt
python -V && nvidia-smi -L && $test_tracktor with tracktor.tracker.boxes_enlargement_factor=1.0 tracktor.tracker.do_reid=True tracktor.tracker.do_align=False |& tee drive_output/scheduled_training/2.0_model_tests/reid_noalign_1.0.txt
python -V && nvidia-smi -L && $test_tracktor with tracktor.tracker.boxes_enlargement_factor=1.0 tracktor.tracker.do_reid=False tracktor.tracker.do_align=True |& tee drive_output/scheduled_training/2.0_model_tests/noreid_align_1.0.txt
python -V && nvidia-smi -L && $test_tracktor with tracktor.tracker.boxes_enlargement_factor=1.0 tracktor.tracker.do_reid=True tracktor.tracker.do_align=True |& tee drive_output/scheduled_training/2.0_model_tests/reid_align_1.0.txt
