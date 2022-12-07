#!/bin/bash

dos2unix run_flow.sh
dos2unix run_2_step_flow.sh
echo FORDOR opt-2.7b1
export CUDA_VISIBLE_DEVICES=2
./run_flow.sh --model_name=facebook/opt-2.7b --output_dir=attn1_opt-2.7b_32l --num_examples=12000 --method=attn1 > opt-2.7b_attn1.log &
echo FORDOR opt-2.7b2
export CUDA_VISIBLE_DEVICES=3
./run_flow.sh --model_name=facebook/opt-2.7b --output_dir=attn2_opt-2.7b_32l --num_examples=12000 --method=attn2 > opt-2.7b_attn2.log &
echo FORDOR opt-2.7b3
export CUDA_VISIBLE_DEVICES=4
./run_flow.sh --model_name=facebook/opt-2.7b --output_dir=attn3_opt-2.7b_32l --num_examples=12000 --method=attn3 > opt-2.7b_attn3.log &
#echo FORDOR opt-2.7b4
#export CUDA_VISIBLE_DEVICES=4
#./run_flow.sh --model_name=facebook/opt-2.7b --output_dir=perplexity_opt-2.7b_12k_2 --num_examples=12000 --method=perplexity > opt-2.7b_test4_pp.log &
echo done