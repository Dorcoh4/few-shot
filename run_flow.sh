#!/bin/bash
if [[ $# -lt 2 ]]; then
    echo "Illegal number of parameters" >&2
    exit 2
fi

MODEL_NAME=$1
OUTPUT_DIR=$2
NUM_EXAMPLES=${3-12000}
PROMPT=${4-"Is this sentence common?"}
HIGH_PP=${5-"no"}
LOW_PP=${6-"yes"}
SHOT=${7-0}
PROMPT_AFTER=$8

echo "Model name:" $MODEL_NAME
echo "Output dir:" $OUTPUT_DIR
echo "Prompt:" $PROMPT
echo "After prompt:" $PROMPT_AFTER
echo "high perplexity answer:" $HIGH_PP
echo "low perplexity answer:" $LOW_PP
echo "shot:" $SHOT
echo "dataset size:" $NUM_EXAMPLES
mkdir $OUTPUT_DIR
set -e
python main.py --model_name=$MODEL_NAME --output_dir=$OUTPUT_DIR --num_examples=$NUM_EXAMPLES
echo "quantiles finished"
python save_examples.py --model_name=$MODEL_NAME --output_dir=$OUTPUT_DIR --num_examples=$NUM_EXAMPLES
echo "saved examples"
python perplex.py --model_name=$MODEL_NAME --output_dir=$OUTPUT_DIR --prompt_q="$PROMPT" --high_pp_target="$HIGH_PP" --low_pp_targe\
t="$LOW_PP" --shot=$SHOT --num_examples=$NUM_EXAMPLES --prompt_after="$PROMPT_AFTER"
echo "donso"