#!/bin/bash

game=${1}
cfg=${2}
train_dir=${3}
iterations=${4}
more_options=(${@:5})
if [[ -z $game ]] || [[ ! -f $cfg ]]; then
    echo "Usage: $0 GAME CFG_FILE TRAIN_DIR ITERATIONS [MORE_OPTIONS]..."
    echo "Example: $0 go cfg/go.cfg go_9x9_gmz 300"
    echo "Supported games: atari go gomoku"
    echo "Available configs: $(ls cfg/*.cfg | xargs -n1 basename | tr '\n' ' ')"
    exit 1
fi
if [[ -z $container ]]; then
    echo "Please run this script in the container launched by 'scripts/start-container.sh'"
    exit 1
fi

tools/quick-run.sh train $game $cfg $iterations -n $train_dir ${more_options[@]}
latest_model=$(find $train_dir/model -name "*.pkl" | sort -V | tail -n1)
[ -e $latest_model ] &&
    PYTHONPATH=. python tools/recreate_model.py $game <(sed "s/decoder_output_at_inference=false/decoder_output_at_inference=true/" $cfg) $latest_model >/dev/null 2>&1 &&
    mv ${latest_model%.pkl}_fixed.pt ${latest_model%.pkl}.pt || echo $(tput setaf 1)"Failed to export model from $train_dir"$(tput sgr0)
