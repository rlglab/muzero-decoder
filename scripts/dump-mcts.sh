#!/bin/bash

game=${1}
cfg=${2}
model=${3}
output=${4}
more_options=(${@:5})
if [[ -z $game ]] || [[ -z $cfg ]] || [[ -z $model ]] || [[ -z $output ]]; then
    echo "Usage: $0 GAME CFG MODEL OUTPUT_DIR [MORE_OPTIONS]..."
    echo "Example: $0 go cfg/go.cfg go_9x9_gmz/model/weight_iter_60000.pt go_9x9_gmz/mcts_dump"
    exit 1
fi
if [[ -z $container ]]; then
    echo "Please run this script in the container launched by 'scripts/start-container.sh'"
    exit 1
fi
dpkg -l | grep -q graphviz || { apt -y update && apt -y install graphviz && pip install graphviz || exit 1; }

num_games=1
dump_opts="actor_dump_mcts_tree=true:zero_num_threads=1:zero_num_parallel_games=1:zero_training_directory=$output"
if [[ ${more_options[@]} == *" -conf_str "* ]]; then
    i=0
    while [[ ${more_options[$i]} != -conf_str ]]; do ((i++)); done
    more_options[$i+1]+=:$dump_opts
else
    more_options+=(-conf_str $dump_opts)
fi
mkdir -p $output
tools/run-selfplay.sh $game $model $cfg $output/selfplay.sgf $num_games ${more_options[@]}
for file in $output/{tree_dump,move_dump,illegal_dump,terminal_dump}.txt; do
    nlim=$(grep -n -m1 game${num_games}_ $file | cut -d: -f1)
    [[ $nlim ]] && sed -i "${nlim},\$d" $file
done
mkdir -p $output/tree_dump
PYTHONPATH=. python tools/visualize-tree.py $output/tree_dump < $output/tree_dump.txt
if [[ $game != atari ]]; then
    mkdir -p $output/move_dump
    build/${game}/minizero_${game} -mode plotter -conf_file $cfg -conf_str nn_file_name=$model:decoder_out_file_path=$output/move_dump < $output/move_dump.txt
fi
