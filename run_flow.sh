#!/bin/bash
#if [[ $# -lt 2 ]]; then
#    echo "Illegal number of parameters" >&2
#    exit 2
#fi
#
#MODEL_NAME=$1
#OUTPUT_DIR=$2
#NUM_EXAMPLES=${3-12000}
#PROMPT=${4-"Is this sentence common?"}
#HIGH_PP=${5-"no"}
#LOW_PP=${6-"yes"}
#SHOT=${7-0}
#PROMPT_AFTER=$8
#
#echo "Model name:" $MODEL_NAME
#echo "Output dir:" $OUTPUT_DIR
#echo "Prompt:" $PROMPT
#echo "After prompt:" $PROMPT_AFTER
#echo "high perplexity answer:" $HIGH_PP
#echo "low perplexity answer:" $LOW_PP
#echo "shot:" $SHOT
#echo "dataset size:" $NUM_EXAMPLES
#mkdir $OUTPUT_DIR
set -e
python main.py "$@"
echo "quantiles finished"
python save_examples.py "$@"
echo "saved examples"
python perplex.py "$@"
echo "donso"