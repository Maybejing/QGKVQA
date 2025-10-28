#!/bin/bash

torchrun --nproc_per_node=2 --master_port 29511 main.py --data_name fvqa_sp0 --cfg ht_fvqa --n_hop 1 --num_workers 20 --lr 0.0001 --wd 0.00001 --exp_name ht_fvqa_sp0 --abl_ans_fc




