#!/bin/bash

dos2unix run_flow.sh
echo FORDOR tpp
./run_flow.sh  bigscience/T0pp tpp_50k 50000 > tpp_50k.log
echo FORDOR tp
./run_flow.sh  bigscience/T0p tp_50k 50000 > tp_50k.log
echo FORDOR t0
./run_flow.sh  bigscience/T0 t0_50k 50000 > t0_50k.log
echo FORDOR tsp
./run_flow.sh  bigscience/T0_single_prompt tsp_50k 50000 > tsp_50k.log
echo FORDOR toto
./run_flow.sh  bigscience/T0_original_task_only toto_50k 50000 > toto_50k.log
echo FORDOR tpp short
python perplex.py --model_name=bigscience/T0pp --output_dir=tpp_50k --shot=6 --num_examples=50000 > tpp_50k_short.log
echo FORDOR tp short
python perplex.py --model_name=bigscience/T0p --output_dir=tp_50k --shot=6 --num_examples=50000 > tp_50k_short.log
echo FORDOR t0 short
python perplex.py --model_name=bigscience/T0 --output_dir=t0_50k --shot=6 --num_examples=50000 > t0_50k_short.log
echo FORDOR tsp short
python perplex.py --model_name=bigscience/T0_single_prompt --output_dir=tsp_50k --shot=6 --num_examples=50000 > tsp_50k_short.log
echo FORDOR toto short
python perplex.py --model_name=bigscience/T0_original_task_only --output_dir=toto_50k --shot=6 --num_examples=50000 > toto_50k_short.log
echo done

