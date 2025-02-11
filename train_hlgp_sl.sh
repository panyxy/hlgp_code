#!/bin/bash


export RESULTS_DIR=./results/hlgp_sl

cd insertion
make
cd ..

cd graph_partition/hlgp_sl/
python train.py \
--log_path ${RESULTS_DIR} \
--trainset_size 100 \
--batch_size 50 \
--beam_size 16 \
--problem_size 1000 \
--steps_per_epoch 500 \
--n_epochs 20 \
--lr 0.00001 \
--lr_decay 0.9 \
--lr_decay_step 5 \
--starting_epoch 1 \
--perm_device 0 \
--glb_part_device 0 \
--loc_part_device 0 \
--valset_size 100 \
--eval_batch_size 50 \
--eval_beam_size 1 \
--eval_problem_size 1000 \
--train_revision_iters 5 \
--eval_revision_iters 5 \
--n_neigh 2 \
--revise_direction forward \
--knns 250 \
--use_local_policy \
--train_part_model \
--use_eval \






