if [[ $# -ne 2 ]]; then
    echo "Illegal number of parameters" >&2
    exit 2
fi

MODEL_NAME=$1
OUTPUT_DIR=$2
echo "Model name:" $MODEL_NAME
echo "Output dir:" $OUTPUT_DIR
mkdir $OUTPUT_DIR
set -e
python main.py --model_name=$MODEL_NAME --output_dir=$OUTPUT_DIR
echo "quantiles finished"
python save_examples.py --model_name=$MODEL_NAME --output_dir=$OUTPUT_DIR
echo "saved examples"
python perplex.py --model_name=$MODEL_NAME --output_dir=$OUTPUT_DIR
echo "donso"