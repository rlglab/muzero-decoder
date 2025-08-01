#!/bin/bash

game=${1}
cfg=${2}
model=${3}
input=${4}
output=${5}
if [[ -z $game ]] || [[ -z $cfg ]] || [[ -z $model ]] || [[ -z $input ]] || [[ -z $output ]]; then
    echo "Usage: $0 GAME CFG MODEL INPUT_SGF[:LINE] OUTPUT_DIR"
    echo "Example: $0 go cfg/go.cfg go_9x9_gmz/model/weight_iter_60000.pt go_9x9_gmz/model/sgf/300.sgf go_9x9_gmz/unroll_analysis"
    exit 1
fi
if [[ -z $container ]]; then
    echo "Please run this script in the container launched by 'scripts/start-container.sh'"
    exit 1
fi

line=1
if [[ $input == *.sgf:* ]]; then
    line=${input##*:}
    input=${input%:*}
fi

mkdir -p $output
PYTHONPATH=. python tools/unroll_plot_tool.py -game $game -conf_file <(sed "s/decoder_output_at_inference=false/decoder_output_at_inference=true/" $cfg) -nn_file_name $model -sgf $input -l $line -out_dir $output
