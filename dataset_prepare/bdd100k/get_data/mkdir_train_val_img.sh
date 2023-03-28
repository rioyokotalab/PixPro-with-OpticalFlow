#!/bin/bash
script_dir=$(
    cd "$(dirname "$0")" || exit
    pwd
)
bdd100k_root=$(
    cd "$1" || exit
    pwd
)

pushd "$bdd100k_root"

# データの置き場所に応じて変える
dirPath="$bdd100k_root/bdd100k/videos"
resPath="$bdd100k_root/bdd100k/images"

subsets=`find ${dirPath}/* -maxdepth 0 -type d | xargs basename -a`

mkdir -p ${resPath}
logdir="$script_dir"/log_mkdir
mkdir -p "$logdir"

for subset in ${subsets};
do
    echo start processing ${subset}...
    mkdir -p ${resPath}/${subset}
    mkjob_name="$logdir"/mkdir_${subset}.log
    find ${dirPath}/${subset}/ -maxdepth 1 -type f -name "*.mov" | parallel --joblog "$mkjob_name" mkdir -p ${resPath}/${subset}/{/.}
done

popd
