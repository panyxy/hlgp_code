#!/bin/bash

export RESULTS_DIR=./results/hlgp_rl

cd insertion
make
cd ..

cd graph_partition/hlgp_rl/
python train.py \
--log_path ${RESULTS_DIR} \
--batch_size 5 \
--beam_size 20 \
--problem_size 1000 \
--steps_per_epoch 256 \
--n_epochs 20 \
--gnn_topK 100 \
--starting_epoch 1 \
--perm_device 0 \
--glb_part_device 0 \
--loc_part_device 0 \
--loc_batch_size 2 \
--loc_beam_size 20 \
--valset_size 100 \
--eval_batch_size 20 \
--eval_problem_size 1000 \
--train_glb_revision_iters 1 \
--train_loc_revision_iters 5 \
--eval_revision_iters 5 \
--n_neigh 2 \
--revise_direction forward \
--glb_lambda 0.1 \
--loc_lambda 0.005 \
--no_subp_epoch 2 \
--use_local_policy \
--train_local_part \
--train_global_part \
--use_eval \



