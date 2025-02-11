#!/bin/bash



export RESULTS_DIR=./results/hlgp_rl
export MODEL_PATH=../../pretrained/hlgp_rl


cd graph_partition/hlgp_rl/
python eval.py \
--log_path $RESULTS_DIR \
--glb_model_load_path ${MODEL_PATH} \
--loc_model_load_path ${MODEL_PATH} \
--revision_lens 20 \
--revision_iters 5 \
--batch_size ${1} \
--valset_size 100 \
--problem_size ${2} \
--gnn_topK ${3} \
--perm_model_type am_reviser \
--perm_device 0 \
--glb_part_device 0 \
--loc_part_device 0 \
--eval_revision_iters ${4} \
--revise_direction forward \
--use_local_policy \
--use_subp_eval 1 \
--dataset_path ../../data/vrp/vrp${5}_test_seed1234.pkl \







