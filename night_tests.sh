#!/bin/bash

dos2unix run_flow.sh
dos2unix run_2_step_flow.sh
echo FORDOR tpp2
#./run_2_step_flow.sh  bigscience/T0pp tpp_50k 50000 "Does any word in this sentence surprise you?" "yes" "no" > tpp_50k2.log
echo FORDOR tpp3
python perplex.py --model_name=bigscience/T0pp --output_dir=tpp_50k --shot=2 --prompt_q="Does any word in this sentence surprise you?" --high_pp_target=yes --low_pp_target=no --num_examples=50000 > tpp_50k_short1.log
echo FORDOR tpp4
python perplex.py --model_name=bigscience/T0pp --output_dir=tpp_50k --prompt_q="Does any word in the following sentence surprise you?" --high_pp_target=yes --low_pp_target=no --num_examples=50000 > tpp_50k_short2.log
echo FORDOR tpp5
python perplex.py --model_name=bigscience/T0pp --output_dir=tpp_50k --shot=2 --prompt_q="Does any word in the following sentence surprise you?" --high_pp_target=yes --low_pp_target=no --num_examples=50000 > tpp_50k_short3.log
echo FORDOR tpp6
python perplex.py --model_name=bigscience/T0pp --output_dir=tpp_50k --prompt_q="Read the sentence then answer the following question." --prompt_after="Did any of the words in the previous sentence surprise you?" --high_pp_target=yes --low_pp_target=no --num_examples=50000 > tpp_50k_short4.log
echo FORDOR tpp7
python perplex.py --model_name=bigscience/T0pp --output_dir=tpp_50k --prompt_q="Read the sentence then answer the following question." --prompt_after="Would you have chosen those words?" --high_pp_target=no --low_pp_target=yes --num_examples=50000 > tpp_50k_short5.log
echo FORDOR tpp8
python perplex.py --model_name=bigscience/T0pp --output_dir=tpp_50k --shot=2 --prompt_q="Read the sentence then answer the following question." --prompt_after="Did any of the words in the previous sentence surprise you?" --high_pp_target=yes --low_pp_target=no --num_examples=50000 > tpp_50k_short6.log
echo FORDOR tpp9
python perplex.py --model_name=bigscience/T0pp --output_dir=tpp_50k --shot=2 --prompt_q="Read the sentence then answer the following question." --prompt_after="Would you have chosen those words?" --high_pp_target=no --low_pp_target=yes --num_examples=50000 > tpp_50k_short7.log
echo FORDOR tpp10
python perplex.py --model_name=bigscience/T0pp --output_dir=tpp_50k --prompt_q="Read the sentence then answer the following question." --prompt_after="Was the sentence predictable?" --high_pp_target=no --low_pp_target=yes --num_examples=50000 > tpp_50k_short8.log
echo FORDOR tpp11
python perplex.py --model_name=bigscience/T0pp --output_dir=tpp_50k --shot=2 --prompt_q="Read the sentence then answer the following question." --prompt_after="Was the sentence predictable?" --high_pp_target=no --low_pp_target=yes --num_examples=50000 > tpp_50k_short9.log
echo FORDOR tpp12
python perplex.py --model_name=bigscience/T0pp --output_dir=tpp_50k --prompt_q="" --prompt_after="Was the previous sentence predictable?" --high_pp_target=no --low_pp_target=yes --num_examples=50000 > tpp_50k_short10.log
echo FORDOR tpp13
python perplex.py --model_name=bigscience/T0pp --output_dir=tpp_50k --prompt_q="" --prompt_after="Was the previous sentence common?" --high_pp_target=no --low_pp_target=yes --num_examples=50000 > tpp_50k_short11.log
echo FORDOR tpp14
python perplex.py --model_name=bigscience/T0pp --output_dir=tpp_50k --shot=2 --prompt_q="" --prompt_after="Was the previous sentence common?" --high_pp_target=no --low_pp_target=yes --num_examples=50000 > tpp_50k_short12.log
echo FORDOR tpp15
python perplex.py --model_name=bigscience/T0pp --output_dir=tpp_50k --prompt_q="Is this sentence unusual?" --high_pp_target=yes --low_pp_target=no --num_examples=50000 > tpp_50k_short13.log
echo FORDOR tpp16
python perplex.py --model_name=bigscience/T0pp --output_dir=tpp_50k --prompt_q="Are the words in this sentence predictable?" --high_pp_target=yes --low_pp_target=no --num_examples=50000 > tpp_50k_short14.log
echo FORDOR tpp17
python perplex.py --model_name=bigscience/T0pp --output_dir=tpp_50k --shot=2 --prompt_q="Are the words in this sentence predictable?" --high_pp_target=no --low_pp_target=yes --num_examples=50000 > tpp_50k_short15.log
echo FORDOR tpp18
python perplex.py --model_name=bigscience/T0pp --output_dir=tpp_50k --shot=2 --prompt_q="Is this sentence unusual?" --high_pp_target=yes --low_pp_target=no --num_examples=50000 > tpp_50k_short16.log
echo FORDOR tpp19
python perplex.py --model_name=bigscience/T0pp --output_dir=tpp_50k --prompt_q="Read the words in the following sentence then answer the question." --prompt_after="After reading the first word did the rest of the words seem natural to you?" --high_pp_target=no --low_pp_target=yes --num_examples=50000 > tpp_50k_short17.log
echo FORDOR tpp20
python perplex.py --model_name=bigscience/T0pp --output_dir=tpp_50k --shot=2 --prompt_q="Read the words in the following sentence then answer the question." --prompt_after="After reading the first word did the rest of the words seem natural to you?" --high_pp_target=no --low_pp_target=yes --num_examples=50000 > tpp_50k_short18.log
echo FORDOR tpp21
python perplex.py --model_name=bigscience/T0pp --output_dir=tpp_50k --prompt_q="Read the following sentence then answer the question." --prompt_after="Was the sentence unique?" --high_pp_target=yes --low_pp_target=no --num_examples=50000 > tpp_50k_short19.log
echo FORDOR tpp22
python perplex.py --model_name=bigscience/T0pp --output_dir=tpp_50k --prompt_q="Read the following sentence then answer the question." --prompt_after="Was the sentence unique?" --high_pp_target=yes --low_pp_target=no --num_examples=50000 > tpp_50k_short20.log
echo FORDOR tpp23
python perplex.py --model_name=bigscience/T0pp --output_dir=tpp_50k --shot=2 --prompt_q="Read the following sentence then answer the question." --prompt_after="Was the sentence unique?" --high_pp_target=no --low_pp_target=yes --num_examples=50000 > tpp_50k_short21.log
echo FORDOR tpp24
python perplex.py --model_name=bigscience/T0pp --output_dir=tpp_50k  --prompt_q="Is you training loss relatively high on this sentence?" --high_pp_target=yes --low_pp_target=no --num_examples=50000 > tpp_50k_short22.log
echo FORDOR tpp25
python perplex.py --model_name=bigscience/T0pp --output_dir=tpp_50k --shot=2 --prompt_q="Is you training loss relatively high on this sentence?" --high_pp_target=yes --low_pp_target=no --num_examples=50000 > tpp_50k_short23.log
echo FORDOR tpp26
python perplex.py --model_name=bigscience/T0pp --output_dir=tpp_50k  --prompt_q="Is you training loss relatively low on this sentence?" --high_pp_target=no --low_pp_target=yes --num_examples=50000 > tpp_50k_short24.log
echo FORDOR tpp27
python perplex.py --model_name=bigscience/T0pp --output_dir=tpp_50k --shot=2 --prompt_q="Is you training loss relatively low on this sentence?" --high_pp_target=no --low_pp_target=yes --num_examples=50000 > tpp_50k_short25.log





echo done

