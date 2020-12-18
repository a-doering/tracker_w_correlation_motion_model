#!/bin/bash

test_tracktor='python /content/tracker_w_correlation_motion_model/experiments/scripts/test_tracktor.py'

rm -rf output/debug_images/*

python -V && nvidia-smi -L && $test_tracktor with tracktor.tracker.boxes_enlargement_factor=1.0 tracktor.tracker.do_reid=False tracktor.tracker.do_align=False |& tee drive_output/align_fix/1.5_model_tests/noreid_noalign_1.0/noreid_noalign_1.0_1.5_model.txt
#mv output/debug_images/MOT17-13 drive_output/align_fix/1.5_model_tests/noreid_noalign_1.0/
python -V && nvidia-smi -L && $test_tracktor with tracktor.tracker.boxes_enlargement_factor=1.0 tracktor.tracker.do_reid=False tracktor.tracker.do_align=True |& tee drive_output/align_fix/1.5_model_tests/noreid_align_1.0/noreid_align_1.0_1.5_model.txt
#mv output/debug_images/MOT17-13 drive_output/align_fix/1.5_model_tests/noreid_align_1.0/
python -V && nvidia-smi -L && $test_tracktor with tracktor.tracker.boxes_enlargement_factor=1.0 tracktor.tracker.do_reid=True tracktor.tracker.do_align=False |& tee drive_output/align_fix/1.5_model_tests/reid_noalign_1.0/reid_noalign_1.0_1.5_model.txt
#mv output/debug_images/MOT17-13 drive_output/align_fix/1.5_model_tests/reid_noalign_1.0/
python -V && nvidia-smi -L && $test_tracktor with tracktor.tracker.boxes_enlargement_factor=1.0 tracktor.tracker.do_reid=True tracktor.tracker.do_align=True |& tee drive_output/align_fix/1.5_model_tests/reid_align_1.0/reid_align_1.0_1.5_model.txt
#mv output/debug_images/MOT17-13 drive_output/align_fix/1.5_model_tests/reid_align_1.0/


python -V && nvidia-smi -L && $test_tracktor with tracktor.tracker.boxes_enlargement_factor=1.5 tracktor.tracker.do_reid=False tracktor.tracker.do_align=False |& tee drive_output/align_fix/1.5_model_tests/noreid_noalign_1.5/noreid_noalign_1.5_1.5_model.txt
#mv output/debug_images/MOT17-13 drive_output/align_fix/1.5_model_tests/noreid_noalign_1.5/
python -V && nvidia-smi -L && $test_tracktor with tracktor.tracker.boxes_enlargement_factor=1.5 tracktor.tracker.do_reid=False tracktor.tracker.do_align=True |& tee drive_output/align_fix/1.5_model_tests/noreid_align_1.5/noreid_align_1.5_1.5_model.txt
#mv output/debug_images/MOT17-13 drive_output/align_fix/1.5_model_tests/noreid_align_1.5/
python -V && nvidia-smi -L && $test_tracktor with tracktor.tracker.boxes_enlargement_factor=1.5 tracktor.tracker.do_reid=True tracktor.tracker.do_align=False |& tee drive_output/align_fix/1.5_model_tests/reid_noalign_1.5/reid_noalign_1.5_1.5_model.txt
#mv output/debug_images/MOT17-13 drive_output/align_fix/1.5_model_tests/reid_noalign_1.5/
python -V && nvidia-smi -L && $test_tracktor with tracktor.tracker.boxes_enlargement_factor=1.5 tracktor.tracker.do_reid=True tracktor.tracker.do_align=True |& tee drive_output/align_fix/1.5_model_tests/reid_align_1.5/reid_align_1.5_1.5_model.txt
#mv output/debug_images/MOT17-13 drive_output/align_fix/1.5_model_tests/reid_align_1.5/


python -V && nvidia-smi -L && $test_tracktor with tracktor.tracker.boxes_enlargement_factor=2.0 tracktor.tracker.do_reid=False tracktor.tracker.do_align=False |& tee drive_output/align_fix/1.5_model_tests/noreid_noalign_2.0/noreid_noalign_2.0_1.5_model.txt
#mv output/debug_images/MOT17-13 drive_output/align_fix/1.5_model_tests/noreid_noalign_2.0/
python -V && nvidia-smi -L && $test_tracktor with tracktor.tracker.boxes_enlargement_factor=2.0 tracktor.tracker.do_reid=False tracktor.tracker.do_align=True |& tee drive_output/align_fix/1.5_model_tests/noreid_align_2.0/noreid_align_2.0_1.5_model.txt
#mv output/debug_images/MOT17-13 drive_output/align_fix/1.5_model_tests/noreid_align_2.0/
python -V && nvidia-smi -L && $test_tracktor with tracktor.tracker.boxes_enlargement_factor=2.0 tracktor.tracker.do_reid=True tracktor.tracker.do_align=False |& tee drive_output/align_fix/1.5_model_tests/reid_noalign_2.0/reid_noalign_2.0_1.5_model.txt
#mv output/debug_images/MOT17-13 drive_output/align_fix/1.5_model_tests/reid_noalign_2.0/
python -V && nvidia-smi -L && $test_tracktor with tracktor.tracker.boxes_enlargement_factor=2.0 tracktor.tracker.do_reid=True tracktor.tracker.do_align=True |& tee drive_output/align_fix/1.5_model_tests/reid_align_2.0/reid_align_2.0_1.5_model.txt
#mv output/debug_images/MOT17-13 drive_output/align_fix/1.5_model_tests/reid_align_2.0/

