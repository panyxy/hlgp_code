#!/bin/bash

export RESULTS_DIR=./results/hlgp_sl
export MODEL_PATH=../../pretrained/hlgp_sl



cd graph_partition/hlgp_sl/
python eval.py \
--log_path $RESULTS_DIR \
--batch_size ${1} \
--beam_size ${2} \
--problem_size ${3} \
--valset_size 100 \
--problem_type cvrp \
--revision_lens 20 \
--revision_iters 5 \
--knns ${4} \
--perm_decode_type greedy \
--perm_model_type am_reviser \
--perm_device 0 \
--glb_part_device 0 \
--loc_part_device 0 \
--model_load_path ${MODEL_PATH} \
--revise_direction forward \
--eval_revision_iters ${5} \
--use_local_policy \
--dataset_path ../../data/vrp/vrp${6}_test_seed1234.pkl \

