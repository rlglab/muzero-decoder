#!/bin/bash
usage() {
    cat << USAGE
Usage: $0 GAME_TYPE NN_FILE_NAME CONF_FILE SGF_OUTPUT NUM_GAMES [OPTION]...

  -sgf_file  set output file
  -conf_str  set config string
  -g         set available GPU
USAGE
}

game=$1
nn_file_name=$2
conf_file=$3
sgf_file=$4
num_games=$5

conf_str=nn_file_name=$nn_file_name:program_auto_seed=true:zero_num_threads=8:zero_num_parallel_games=32:zero_actor_intermediate_sequence_length=0
executable=build/${game}/minizero_${game}

if ! shift 5; then
    usage
    exit 1
fi

while [[ $1 ]]; do
    opt=$1
    shift
    case "$opt" in
    -g)
        CUDA_VISIBLE_DEVICES=$(<<< $1 grep -o . | xargs | tr ' ' ',')
        shift
        ;;
    -sgf_file)
        sgf_file=$1
        shift
        ;;
    -conf_str)
        conf_str+=:$1
        shift
        ;;
    *)
        usage
        [[ $opt == -h ]]
        exit $?
        ;;
    esac
done

for file in $executable $nn_file_name $conf_file; do
    if [ ! -e "$file" ]; then
        echo ${file} not found >&2
        exit 1
    fi
done

if [[ -e $sgf_file ]]; then
    echo "SGF file $sgf_file ($(wc -l < $sgf_file) records) already exists!"
    read -n1 -p "Continue? [Y/n] " option
    echo
    [[ ${option,,} == n ]] && exit 0
else
    touch $sgf_file
fi

if (( $(wc -l < $sgf_file) < $num_games )); then
    coproc SELFPLAY { CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES $executable -mode sp -conf_file $conf_file -conf_str "$conf_str" 2>${sgf_file%.sgf}.log; }
    trap 'kill $SELFPLAY_PID 2>/dev/null' EXIT
    trap 'exit 127' TERM INT

    echo -n "Generating $((num_games - $(wc -l < $sgf_file))) self-play games "
    echo start >&${SELFPLAY[1]}
    while ps -p $SELFPLAY_PID >/dev/null && IFS= read -r selfplay <&${SELFPLAY[0]}; do
        if (( $(wc -l < $sgf_file) < $num_games )); then
            echo -n .
            echo $selfplay | grep "#" | sed -E "s/^.+\(/(/" >> $sgf_file
        else
            echo quit >&${SELFPLAY[1]}
        fi
    done
    
    if (( $(wc -l < $sgf_file) >= $num_games )); then
        echo " done!"
        echo "Saved as $sgf_file"
    else
        echo " failed!"
        exit 1
    fi
    trap - EXIT
fi
